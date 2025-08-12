#!/bin/bash

#!/bin/bash

JOBSTARTDATE=$(date)

OFFSET=$1
STRIDE=$2
SAMPLE_NAME=$3
INPUTSTEM=$4
INPUTLIST=$5
FILEIDLIST=$6 # make this using check_files.py

echo "Inputlist: ${INPUTLIST}"
echo "File ID list: ${FILEIDLIST}"

# we assume we are already in the container
export OMP_NUM_THREADS=16

# Location of the repo being used
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/data_prep/tuftscluster/
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/

# Set below for debug
#SLURM_ARRAY_TASK_ID=0

# Parameters for shower-keypoint retraining and reco-retuning
#RECOVER=v3dev_reco_retune
#UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/
#LARMATCH_DIR=${UBDL_DIR}/larflow/larmatchnet/larmatch/
#WEIGHTS_DIR=${LARMATCH_DIR}/checkpoints/easy-wave-79/
#WEIGHT_FILE=checkpoint.93000th.tar
#CONFIG_FILE=${WORKDIR}/config_larmatchme_deploycpu.yaml
#LARMATCHME_SCRIPT=${LARMATCH_DIR}/deploy_larmatchme_v2.py

# More common parameters dependent on version-specific variables
#RECO_TEST_DIR=${UBDL_DIR}/larflow/larflow/Reco/test/
OUTPUT_DIR=${WORKDIR}/outdir/${SAMPLE_NAME}/
OUTPUT_LOGDIR=${WORKDIR}/logdir/${RECOVER}/${SAMPLE_NAME}/

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_LOGDIR

# WE WANT TO RUN MULTIPLE FILES PER JOB IN ORDER TO BE GRID EFFICIENT
start_jobid=$(( ${OFFSET} + ${SLURM_ARRAY_TASK_ID}*${STRIDE}  ))

#echo "JOB ARRAYID: ${SLURM_ARRAY_TASK_ID} -- CUDA DEVICES: ${CUDA_VISIBLE_DEVICES}"
#let ndevices=$(echo $CUDA_VISIBLE_DEVICES | sed 's|,| |g' | wc -w )
#let devnum=$(expr $SLURM_ARRAY_TASK_ID % $ndevices + 1)
#cudaid=$(echo $CUDA_VISIBLE_DEVICES | sed 's|,| |g' | awk '{print '"\$${devnum}"'}')
#cudadev=$(echo "cuda:${cudaid}")
cudadev="cpu"
echo "JOB ARRAYID: ${SLURM_ARRAY_TASK_ID} : CUDA DEVICE = ${cudadev} : NODE = ${SLURMD_NODENAME}"

# LOCAL JOBDIR
local_jobdir=`printf /tmp/flashmatch_dataprep_jobid%04d_${SAMPLE_NAME}_${SLURM_JOB_ID} ${SLURM_ARRAY_TASK_ID}`
rm -rf $local_jobdir
mkdir -p $local_jobdir

# local log file
local_logfile=`printf log_flashmatch_dataprep_${SAMPLE_NAME}_jobid%04d_${SLURM_JOB_ID}.log ${SLURM_ARRAY_TASK_ID}`

#echo "SETUP CONTAINER/ENVIRONMENT"
cd ${UBDL_DIR}
alias python=python3
cd $UBDL_DIR
source setenv_py3_container.sh
source configure_container.sh
#cd ${UBDL_DIR}/larflow/larmatchnet
#source set_pythonpath.sh
#export PYTHONPATH=${LARMATCH_DIR}:${PYTHONPATH}

cd $local_jobdir

echo "STARTING TASK ARRAY ${SLURM_ARRAY_TASK_ID} for ${SAMPLE_NAME}" > ${local_logfile}
echo "running on node $SLURMD_NODENAME" >> ${local_logfile}

#ls /cluster/tufts/wongjiradlab/
#ls /cluster/tufts/wongjiradlabnu/

# run a loop
for ((i=0;i<${STRIDE};i++)); do

    jobid=$(( ${start_jobid} + ${i} ))
    echo "JOBID ${jobid}" >> ${local_logfile}

    # Get fileid from run list
  
    # GET INPUT FILENAME
    let lineno=${jobid}+1
    let fileid=`sed -n ${lineno}p ${FILEIDLIST}`
    let runidlineno=${fileid}+1
    inputfile=`sed -n ${runidlineno}p ${INPUTLIST}`
    baseinput=$(basename $inputfile )
    echo "inputfile path: $inputfile" >> ${local_logfile}
    echo "baseinput: $baseinput" >> ${local_logfile}

    echo "JOBID ${jobid} running FILEID ${fileid} with file: ${baseinput}"

    # define local output file names
    jobname=`printf jobid%04d ${jobid}`
    fileidstr=`printf fileid%04d ${fileid}`
    lm_outfile=$(echo $baseinput  | sed 's|'"${INPUTSTEM}"'|larmatchme_'"${fileidstr}"'|g')
    lm_basename=$(echo $baseinput | sed 's|'"${INPUTSTEM}"'|larmatchme_'"${fileidstr}"'|g' | sed 's|.root||g')
    #baselm=$(echo $baseinput | sed 's|'"${INPUTSTEM}"'|larmatchme_'"${fileidstr}"'|g' | sed 's|.root|_larlite.root|g')
    flashmatch_outfile=$(echo $baseinput  | sed 's|'"${INPUTSTEM}"'|flashmatchdata_'"${fileidstr}"'|g')
    #reco_basename=$(echo $baseinput | sed 's|'"${INPUTSTEM}"'|larflowreco_'"${fileidstr}"'|g' | sed 's|.root||g')
    echo "larmatch outfile : "$lm_basename >> ${local_logfile}
    echo "flashmatch outfile : "$flashmatch_outfile >> ${local_logfile}

    # Copy over input file to be safe
    scp $inputfile $baseinput

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

    $WORKDIR/./../scripts/run_cosmic_reconstruction.sh --input-dlmerged $baseinput --input-larflow $lm_outfile --output test_cosmicreco.root -tb --run-larmatch
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

    ${WORKDIR}/../build/installed/bin/./flashmatch_dataprep --input test_cosmicreco.root --output test_match.root --larcv test_cosmicreco_larcv.root
    # The above will make the output file
    # test_match.root
    
    cp test_match.root $flashmatch_outfile
    
    # copy to subdir in order to keep number of files per folder less than 100. better for file system.
    let nsubdir=${fileid}/100
    subdir=`printf %04d ${nsubdir}`    
    echo "COPY output to "${OUTPUT_DIR}/${subdir}/ >> ${local_logfile}
    mkdir -p $OUTPUT_DIR/${subdir}/    
    cp $flashmatch_outfile ${OUTPUT_DIR}/${subdir}/

    # clean up
    rm ${PWD}/${baseinput}
    rm ${lm_basename}*    
    rm test_cosmicreco*.root
    rm ${flashmatch_outfile}
    rm test_match.root
done

JOBENDDATE=$(date)

echo "Job began at $JOBSTARTDATE" >> $local_logfile
echo "Job ended at $JOBENDDATE" >> $local_logfile

# copy log to logdir
cp $local_logfile $OUTPUT_LOGDIR/

# clean-up
cd /tmp
rm -r $local_jobdir

