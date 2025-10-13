#!/bin/bash

# slurm submission script for running merged dlreco through larmatch and larflowreco
#SBATCH --job-name=flashdata
#SBATCH --output=stdout_estimate_ly.%j.%N.txt
#SBATCH --mem-per-cpu=4000
#SBATCH --time=3-0:00:00
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
##SBATCH --partition=wongjiradlab
##SBATCH --partition=preempt
##SBATCH --exclude=i2cmp006,s1cmp001,s1cmp002,s1cmp003,p1cmp041,c1cmp003,c1cmp004
##SBATCH --gres=gpu:p100:3
##SBATCH --partition ccgpu
##SBATCH --gres=gpu:a100:1
##SBATCH --nodelist=ccgpu01
#SBATCH --error=griderr_estimate_ly.%j.%N.err

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/u20.04_cu111_cudnn8_torch1.9.0_minkowski_npm.sif
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/data_prep/studies/

module load apptainer/1.2.4-suid

cd $WORKDIR

apptainer exec --bind /cluster/tufts/wongjiradlabnu:/cluster/tufts/wongjiradlabnu,/cluster/tufts/wongjiradlab:/cluster/tufts/wongjiradlab ${container} bash -c "cd ${WORKDIR} && source run_gridjob_estimate_ly.sh"
