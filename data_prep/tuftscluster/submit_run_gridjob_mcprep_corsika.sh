#!/bin/bash

# slurm submission script for running merged dlreco through larmatch and larflowreco
#SBATCH --job-name=flashdata
#SBATCH --output=stdout_flashmatch_mcprep_mcc9_v13_bnbnue_corsika_no_anode_throughgoing_sub1.txt
#SBATCH --mem-per-cpu=4000
#SBATCH --time=1-0:00:00
#SBATCH --array=11-246
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
##SBATCH --partition=wongjiradlab
##SBATCH --partition=preempt
##SBATCH --exclude=i2cmp006,s1cmp001,s1cmp002,s1cmp003,p1cmp041,c1cmp003,c1cmp004
##SBATCH --gres=gpu:p100:3
##SBATCH --partition ccgpu
##SBATCH --gres=gpu:a100:1
##SBATCH --nodelist=ccgpu01
#SBATCH --error=griderr_flashmatch_mcprep_mcc9_v13_bnbnue_corsika.%j.%N.no_anode_throughgoing_sub1.err

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/u20.04_cu111_cudnn8_torch1.9.0_minkowski_npm.sif
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/data_prep/tuftscluster/

SAMPLE_NAME=mcc9_v13_bnbnue_corsika
INPUTLIST=${WORKDIR}/mcc9_v13_bnbnue_corsika.list
#FILEIDLIST=${WORKDIR}/runid_mcc9_v13_bnbnue_corsika_20250920.list
FILEIDLIST=${WORKDIR}/runid_mcc9_v13_bnbnue_corsika.list
STRIDE=10
OFFSET=0
INPUTSTEM=larcvtruth

#SAMPLE_NAME=mcc9_v13_bnb_nu_corsika
#INPUTLIST=${WORKDIR}/mcc9_v13_bnb_nu_corsika.list
#FILEIDLIST=${WORKDIR}/runid_mcc9_v13_bnb_nu_corsika.list
#STRIDE=10
#OFFSET=0
#INPUTSTEM=larcv_mctruth
# number of files: 2884
# number of jobs: 288

module load apptainer/1.2.4-suid

cd $WORKDIR

# HDF5 OUTPUT
apptainer exec --bind /cluster/tufts/wongjiradlabnu:/cluster/tufts/wongjiradlabnu,/cluster/tufts/wongjiradlab:/cluster/tufts/wongjiradlab ${container} bash -c "cd ${WORKDIR} && source run_gridjob_hdf5_mcprep_corsika_noanode.sh ${OFFSET} ${STRIDE} ${SAMPLE_NAME} ${INPUTLIST} ${FILEIDLIST} ${INPUTSTEM}"

