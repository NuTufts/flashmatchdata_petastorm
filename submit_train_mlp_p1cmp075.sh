#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=lm_mlp
#SBATCH --output=lightmodel_train_mlp_p1cmp075.log
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000
#SBATCH --time=6-00:00:00
#SBATCH --partition=wongjiradlab
#SBATCH --gres=gpu:p100:1
#SBATCH --error=gridlog_train_mlp.%j.%N.err

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
TRAIN_DIR=/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/ubdl/flashmatchdata_petastorm/
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/ubdl/
BIND_NU=/cluster/tufts/wongjiradlabnu:/cluster/tufts/wongjiradlabnu
BIND_TMP=/tmp:/tmp
#SCRIPT=train_mpl.py
SCRIPT=train_siren.py

module load singularity/3.5.3
cd /cluster/tufts/

# mcc9_v13_bnbnue_corsika: 2000+461 files (train+valid split)
# running 5 files per job:  jobs 0-399 jobs needed for training set
# running 5 files per job:  jobs 400-493
srun singularity exec --nv --bind ${BIND_NU},${BIND_TMP} ${container} bash -c "cd ${UBDL_DIR} && source setenv_py3.sh && source configure.sh && cd ${TRAIN_DIR} && source setenv.sh && python3 ${SCRIPT}"


