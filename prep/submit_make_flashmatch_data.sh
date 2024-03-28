#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=fmdata
#SBATCH --output=fmdata_bnbnue.log
#SBATCH --mem-per-cpu=8000
#SBATCH --time=10:00:00
#SBATCH --array=400-493
##SBATCH --partition=preempt
##SBATCH --partition=wongjiradlab
#SBATCH --partition=batch
#SBATCH --error=gridlog_fmdata_nue.%j.%N.err

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_pyspark.sif
DATA_PREP_DIR=/cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/ubdl/flashmatchdata_petastorm/prep/

module load singularity/3.5.3
cd /cluster/tufts/

# mcc9_v13_bnbnue_corsika: 2000+461 files (train+valid split)
# running 5 files per job:  jobs 0-399 jobs needed for training set
# running 5 files per job:  jobs 400-493
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_flashmatch_data_mcc9_v13_bnbnue_corsika.sh"


