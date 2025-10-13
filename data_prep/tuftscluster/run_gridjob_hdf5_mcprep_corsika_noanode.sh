#!/bin/bash

#!/bin/bash

JOBSTARTDATE=$(date)

OFFSET=$1
STRIDE=$2
SAMPLE_NAME=$3
INPUTLIST=$4
FILEIDLIST=$5
INPUTSTEM=$6

# For debug
#OFFSET=0
#STRIDE=1
#SAMPLE_NAME=mcc9_v13_bnbnue_corsika
#INPUTSTEM=larcvtruth
#INPUTLIST=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/data_prep/tuftscluster/mcc9_v13_bnbnue_corsika.list
#FILEIDLIST=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/data_prep/tuftscluster/runid_mcc9_v13_bnbnue_corsika_20250920.list
#SLURM_ARRAY_TASK_ID=0
#SLURM_JOB_ID=0

echo "Inputlist: ${INPUTLIST}"
echo "File ID list: ${FILEIDLIST}"

# we assume we are already in the container
export OMP_NUM_THREADS=16

# Location of the repo being used
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/data_prep/tuftscluster/
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/
#WORKDIR=/home/twongjirad/working/larbys/gen2/container_u20_env/work/flashmatchdata_petastorm/data_prep/tuftscluster/
#UBDL_DIR=/home/twongjirad/working/larbys/gen2/container_u20_env/work/ubdl/

SSNET_DIR=${UBDL_DIR}/ublarcvserver/networks/uresnet_pytorch/
LIBTORCH_DIR=/usr/local/libtorch1.9.0_cxx11abi/libtorch
LIBTORCH_LIBRARY_DIR=${LIBTORCH_DIR}/lib
LIBTORCH_CMAKE_DIR=${LIBTORCH_DIR}/share/cmake/Torch
LIBTORCH_BIN_DIR=${LIBTORCH_DIR}/bin

# add ssnet folder to pythonpath
[[ ":$PYTHONPATH:" != *":${SSNET_DIR}:"* ]] && export PYTHONPATH="${SSNET_DIR}:${PYTHONPATH}"

# For each file we have 3 steps
# 1. run ssnet on corsika files, copy images, reversing them to tick-forward
# 2. make the fake thrumu image
# 3. run larmatch
# 4. run cosmic reco
# 5. extract the flash-matches using truth-matching

# annoying thing is that ssnet part needs to be production larmatch container
# I guess I can try to run it inside this container

# More common parameters dependent on version-specific variables
OUTPUT_DIR=${WORKDIR}/outdir/no_anode_only_throughgoing/${SAMPLE_NAME}/
OUTPUT_LOGDIR=${WORKDIR}/logdir/no_anode_only_throughgoing/${SAMPLE_NAME}/

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_LOGDIR

# WE WANT TO RUN MULTIPLE FILES PER JOB IN ORDER TO BE GRID EFFICIENT
start_jobid=$(( ${OFFSET} + ${SLURM_ARRAY_TASK_ID}*${STRIDE}  ))

# LOCAL JOBDIR
local_jobdir=`printf /tmp/flashmatch_mcprep_jobid%04d_${SAMPLE_NAME}_${SLURM_JOB_ID} ${SLURM_ARRAY_TASK_ID}`
rm -rf $local_jobdir
mkdir -p $local_jobdir

# local log file
local_logfile=`printf log_flashmatch_dataprep_${SAMPLE_NAME}_jobid%04d_${SLURM_JOB_ID}.log ${SLURM_ARRAY_TASK_ID}`

echo "SETUP CONTAINER/ENVIRONMENT"
# OFF FOR DEBUG
cd ${UBDL_DIR}
source setenv_no_libtorch.sh
source configure_container.sh

cd $local_jobdir

echo "STARTING TASK ARRAY ${SLURM_ARRAY_TASK_ID} for ${SAMPLE_NAME}" > ${local_logfile}
echo "running on node $SLURMD_NODENAME" >> ${local_logfile}

# run a loop
for ((i=0;i<${STRIDE};i++)); do

    jobid=$(( ${start_jobid} + ${i} ))
    echo "JOBID ${jobid}" >> ${local_logfile}

    # Get fileid from run list
  
    # GET INPUT FILENAME
    let lineno=${jobid}+1
    let fileid=`sed -n ${lineno}p ${FILEIDLIST}`
    let runidlineno=${fileid}+1
    inputfile=`sed -n ${runidlineno}p ${INPUTLIST}` # supera file
    inputfile=$(echo $inputfile | sed 's|'"${INPUTSTEM}"'|supera|g') # replace in case inputfile uses larcvtruth
    baseinput=$(basename $inputfile )
    echo "inputfile path: $inputfile" >> ${local_logfile}
    echo "baseinput: $baseinput" >> ${local_logfile}

    opreco_inputpath=$(echo $inputfile | sed 's|supera|opreco|g')
    reco2d_inputpath=$(echo $inputfile | sed 's|supera|reco2d|g')
    larcvtruth_inputpath=$(echo $inputfile | sed 's|supera|'"${INPUTSTEM}"'|g')
    opreco_basename=$(basename $opreco_inputpath)
    reco2d_basename=$(basename $reco2d_inputpath)
    larcvtruth_basename=$(basename $larcvtruth_inputpath)

    echo "JOBID ${jobid} running FILEID ${fileid} with file: ${baseinput}"

    # define local output file names
    jobname=`printf jobid%04d ${jobid}`
    fileidstr=`printf fileid%04d ${fileid}`
    lm_outfile=$(echo $baseinput  | sed 's|supera|larmatchme_'"${fileidstr}"'|g')
    # lm_basename=$(echo $baseinput | sed 's|'"${INPUTSTEM}"'|larmatchme_'"${fileidstr}"'|g' | sed 's|.root||g')
    baselm=$(echo $baseinput | sed 's|supera|larmatchme_'"${fileidstr}"'|g' | sed 's|.root|_larlite.root|g')
    flashmatch_outfile=$(echo $baseinput  | sed 's|supera|flashmatchdata_'"${fileidstr}"'|g' | sed 's|.root|.h5|g')
    echo "larmatch outfile : "$lm_outfile
    # echo "flashmatch outfile : "$flashmatch_outfile >> ${local_logfile}

    # Copy over input file to be safe
    cp $inputfile $baseinput
    cp $opreco_inputpath $opreco_basename
    cp $reco2d_inputpath $reco2d_basename
    cp $larcvtruth_inputpath $larcvtruth_basename
    chmod +w $larcvtruth_basename
    rootcp $baseinput:image2d_wire_tree $larcvtruth_basename
    rootcp $baseinput:chstatus_wire_tree $larcvtruth_basename

    # For corsika files, we need to run ssnet
    python3 ${WORKDIR}/inference_sparse_ssnet_uboone_corsika.py -i $larcvtruth_basename -w $SSNET_DIR/weights -o ssnet_output.root -tb
    python3 ${SSNET_DIR}/recreate_ubspurn.py -i ssnet_output.root -o ssnet_ubspurn_output.root

    hadd -f merged_dlreco_with_ssnet.root ssnet_output.root ssnet_ubspurn_output.root $opreco_basename $reco2d_basename
    rootrm merged_dlreco_with_ssnet.root:larlite_id_tree
    rootcp $opreco_basename:larlite_id_tree merged_dlreco_with_ssnet.root

    # We have to run the cosmic reconstruction
    #     Usage: ./../scripts/run_cosmic_reconstruction.sh [OPTIONS]
    # Run cosmic ray reconstruction using larflow::reco::CosmicParticleReconstruction
    # Required Arguments:
    #   -i, --input-dlmerged FILE    Input dlmerged file (ADC, ssnet, badch images/info)
    #   -l, --input-larflow FILE     Input larflow file (larlite::larflow3dhit objects)
    #   -o, --output FILE            Output file name
    # Optional Arguments:
    #   -n, --num-entries N          Number of entries to process (default: all)
    #   -s, --start-entry N          Starting entry number (default: 0)
    #   -tb, --tick-backwards        Input larcv images are tick-backward (default: false)
    #   -mc, --is-mc                 Store MC information (default: false)
    #   -v, --version N              Reconstruction version (default: 2)
    #   -ll, --log-level N           Log verbosity 0=debug, 1=info, 2=normal, 3=warning, 4=error (default: 1)
    #   --verbose                    Enable verbose output from this script
    #   --run-larmatch               Run larmatch to generate larflow file before cosmic reconstruction
    #   -h, --help                   Display this help message

    $WORKDIR/./../scripts/run_cosmic_reconstruction.sh --input-dlmerged merged_dlreco_with_ssnet.root --input-larflow $lm_outfile --output test_cosmicreco.root --run-larmatch
    # the above will make the following files
    # test_cosmicreco.root
    # test_cosmicreco_larlite.root
    # test_cosmicreco_larcv.root

    # Now run the flashmatch data maker
    # Usage: ../build/installed/bin/./flashmatch_dataprep [OPTIONS]

    # Flash-Track Matching Data Preparation
    # Applies quality cuts and performs flash-track matching on cosmic ray data

    # Required Arguments:
    #   --input FILE              Input ROOT file from cosmic reconstruction
    #   --output FILE             Output ROOT file with matched data

    # Optional Arguments:
    #   --config FILE             Quality cuts configuration (YAML)
    #   --flash-config FILE       Flash matching configuration (YAML)
    #   --debug-output FILE       Debug output file
    #   --max-events N            Maximum number of events to process
    #   --start-event N           Starting event number (default: 0)
    #   --verbosity N             Verbosity level 0-3 (default: 1)
    #   --debug                   Enable debug mode
    #   --no-crt                  Disable CRT matching
    #   --help                    Display this help message

    #   --larcv FILE              LArCV file containing images. Used to make flash prediction.

    # Need to add libtorch temporarily to LD_LIBRARY_PATH
    ORIG_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${LIBTORCH_LIBRARY_DIR}:${ORIG_LD_LIBRARY_PATH}
    #rm ./test_match.h5
    ${WORKDIR}/../build/installed/bin/./flashmatch_mcprep --input test_cosmicreco.root --input-mcinfo merged_dlreco_with_ssnet.root --output-hdf5 test_match.h5 --larcv test_cosmicreco_larcv.root
    # The above will make the output file
    # test_match.h5

    # Restore the old ld_library_path
    export LD_LIBRARY_PATH=${ORIG_LD_LIBRARY_PATH}
    
    cp test_match.h5 $flashmatch_outfile
    
    # copy to subdir in order to keep number of files per folder less than 100. better for file system.
    let nsubdir=${fileid}/100
    subdir=`printf %04d ${nsubdir}`    
    echo "COPY output to "${OUTPUT_DIR}/${subdir}/
    mkdir -p $OUTPUT_DIR/${subdir}/    
    cp $flashmatch_outfile ${OUTPUT_DIR}/${subdir}/

    # clean up
    rm ./*.root
    rm ./test_match.h5
done

JOBENDDATE=$(date)

echo "Job began at $JOBSTARTDATE" >> $local_logfile
echo "Job ended at $JOBENDDATE" >> $local_logfile

# copy log to logdir
cp $local_logfile $OUTPUT_LOGDIR/

# clean-up
cd /tmp
rm -r $local_jobdir

