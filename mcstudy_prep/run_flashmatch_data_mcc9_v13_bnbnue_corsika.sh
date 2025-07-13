#!/bin/bash

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/ubdl/flashmatchdata_petastorm/
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/ubdl/
PYSCRIPT=${WORKDIR}/make_flashmatch_training_data.py

INPUTLIST=${WORKDIR}/prep/nue_corsika_input_filesets.txt

#FILEIDLIST=${WORKDIR}/prep/nue_corsika_fileids.txt
FILEIDLIST=${WORKDIR}/prep/rerun_list_training.txt

#DB_FOLDER=/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v2/
#DB_FOLDER=/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v2_validation/
DB_FOLDER=/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data_v3_training/
FLAGS="--over-write"

export PYTHONPATH=${WORKDIR}:$PYTHONPATH

# stride defines the number of fileids we run per job
stride=5

# jobid assigned by slurm
jobid=${SLURM_ARRAY_TASK_ID}

let portnum=$(expr "${SLURM_ARRAY_TASK_ID}*20+5000")

# calculate the line number we'll get fileids from
let startline=$(expr "${stride}*${jobid}")

jobworkdir=`printf "%s/prep/jobdirs/flashmatch_jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir

local_jobdir=`printf /tmp/flashmatch_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir

# MOVE TO LOCAL JOB DIR
cd $local_jobdir
touch log_jobid${jobid}.txt
local_logfile=`echo ${local_jobdir}/log_jobid${jobid}.txt`

cd $UBDL_DIR
source setenv_py3.sh >> ${local_logfile} 2>&1
source configure.sh >>	${local_logfile} 2>&1
cd $local_jobdir

cp $WORKDIR/sa_5cmvoxels.npz .

CMD="python3 ${PYSCRIPT}"
echo "SCRIPT: ${PYSCRIPT}" >> ${local_logfile} 2>&1
echo "startline: ${startline}" >> ${local_logfile} 2>&1

for i in {1..5}
do
    let lineno=$startline+$i

    let filesetid=`sed -n ${lineno}p ${FILEIDLIST} | awk '{ print $1 }'`
    let fsline=$filesetid+1

    lcvtruth=`sed -n ${fsline}p ${INPUTLIST} | awk '{ print $2 }'`
    opreco=`sed -n ${fsline}p ${INPUTLIST} | awk '{ print $3 }'`
    mcinfo=`sed -n ${fsline}p ${INPUTLIST} | awk '{ print $4 }'`        

    COMMAND="python3 ${PYSCRIPT} -db ${DB_FOLDER} -lcv ${lcvtruth} -mc ${mcinfo} -op ${opreco} --port ${portnum} ${FLAGS}"
    echo $COMMAND
    $COMMAND >> ${local_logfile} 2>&1
    #break
done

cp log_* ${jobworkdir}/

cd /tmp
rm -r $local_jobdir
