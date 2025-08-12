#!/bin/bash

# slurm submission script for running merged dlreco through larmatch and larflowreco
#SBATCH --job-name=flashdata
#SBATCH --output=stdout_mcc9_v29e_dl_run3_G1_extbnb_dlana_sub0.txt
#SBATCH --mem-per-cpu=6000
#SBATCH --time=3-0:00:00
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch,wongjiradlab
##SBATCH --partition=wongjiradlab
##SBATCH --partition=preempt
##SBATCH --exclude=i2cmp006,s1cmp001,s1cmp002,s1cmp003,p1cmp041,c1cmp003,c1cmp004
##SBATCH --gres=gpu:p100:3
##SBATCH --partition ccgpu
##SBATCH --gres=gpu:a100:1
##SBATCH --nodelist=ccgpu01
#SBATCH --error=griderr_lantern_mcc9_v40a_dl_run3b_bnb_nu_overlay_500k_CV_sub00.%j.%N.err

container=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/u20.04_cu111_cudnn8_torch1.9.0_minkowski_npm.sif

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/data_prep/tuftscluster/

SAMPLE_NAME=mcc9_v29e_dl_run3_G1_extbnb_dlana
INPUTSTEM=merged_dlana
INPUTLIST=${WORKDIR}/filelist_mcc9_v29e_dl_run3_G1_extbnb_dlana.txt
FILEIDLIST=${WORKDIR}/runid_mcc9_v29e_dl_run3_G1_extbnb_dlana.txt
STRIDE=20
OFFSET=0
# num files in inputlist: 17687
# with 20 files per job, thats 885 jobs total

module load apptainer/1.2.4-suid

cd $WORKDIR
apptainer exec --bind /cluster/tufts/wongjiradlabnu:/cluster/tufts/wongjiradlabnu,/cluster/tufts/wongjiradlab:/cluster/tufts/wongjiradlab ${container} bash -c "cd ${WORKDIR} && source run_gridjob_dataprep.sh ${OFFSET} ${STRIDE} ${SAMPLE_NAME} ${INPUTSTEM} ${INPUTLIST} ${FILEIDLIST}"

