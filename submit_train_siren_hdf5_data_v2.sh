#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=sirendata
#SBATCH --output=siren_extbnb_unbsinkdiv_w0_0p03.%j.%N.log
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000
#SBATCH --time=6-00:00:00
#SBATCH --partition=wongjiradlab
#SBATCH --gres=gpu:p100:4
#SBATCH --error=griderr_siren_extbnb_unbsinkdiv_w0_0p03.%j.%N.err

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/u20.04_cu111_cudnn8_torch1.9.0_minkowski_npm.sif
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/

module load apptainer/1.2.4-suid
cd /cluster/tufts/
cd $WORKDIR

# mcc9_v13_bnbnue_corsika: 2000+461 files (train+valid split)
# running 5 files per job:  jobs 0-399 jobs needed for training set
# running 5 files per job:  jobs 400-493
#apptainer exec --nv --bind /cluster/tufts:/cluster/tufts ${container} bash -c "cd ${WORKDIR} && source run_train_siren_hdf5_data_v2.sh"
apptainer exec --nv --bind /cluster/tufts:/cluster/tufts ${container} bash -c "cd ${WORKDIR} && source setenv_flashmatchdata.sh && source run_train_siren_hdf5_extbnb_multigpu.sh"


