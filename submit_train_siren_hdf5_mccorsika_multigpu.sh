#!/bin/bash
#SBATCH --job-name=siren-multigpu
#SBATCH --output=logs/siren_multigpu_%j.out
#SBATCH --error=logs/siren_multigpu_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000
#SBATCH --time=6-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --exclude=c[2101-2102]

# Multi-GPU SLURM submission script for SIREN training with HDF5 data
# This script runs distributed training across multiple GPUs on a single node

# Create log directory if it doesn't exist
mkdir -p logs

# Load required modules
module load apptainer/1.2.4-suid

# Set up environment variables for multi-GPU training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export OMP_NUM_THREADS=4

# Container settings
CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/u20.04_cu111_cudnn8_torch1.9.0_minkowski_npm.sif
BIND_OPTS="--bind /cluster/tufts/wongjiradlabnu:/cluster/tufts/wongjiradlabnu"

# Working directory
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm

# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Number of GPUs requested: 4"

# Check GPU availability
singularity exec --nv ${BIND_OPTS} ${CONTAINER} bash -c "nvidia-smi"

# Navigate to working directory and setup environment
cd ${WORKDIR}

# Run training with torchrun for multi-GPU
# Using 4 GPUs on a single node
singularity exec --nv ${BIND_OPTS} ${CONTAINER} bash -c "
    source setenv.sh
    source /cluster/tufts/wongjiradlabnu/twongj01/ubdl-ana/setenv_py3.sh
    source /cluster/tufts/wongjiradlabnu/twongj01/ubdl-ana/configure.sh
    cd ${WORKDIR}

    # Run distributed training
    python -m torch.distributed.run \
        --nproc_per_node=4 \
        --master_addr=\${MASTER_ADDR} \
        --master_port=\${MASTER_PORT} \
        train_siren_hdf5_mccorsika_v2_multigpu.py \
        --config config_siren_hdf5_mccorsika_multigpu.yaml \
        --wandb-project flashmatch-siren-multigpu \
        --wandb-run-name siren-corsika-4gpu-\${SLURM_JOB_ID}
"

echo "Job finished at: $(date)"