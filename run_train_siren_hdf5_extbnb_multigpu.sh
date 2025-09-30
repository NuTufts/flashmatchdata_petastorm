#!/bin/bash

# Interactive script to run multi-GPU SIREN training
# Use this for debugging or interactive sessions

# Set up environment variables for multi-GPU training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export OMP_NUM_THREADS=4

# Number of GPUs to use (adjust based on availability)
NUM_GPUS=4

# Setup environment
source setenv_flashmatchdata.sh

# Print configuration
echo "========================================="
echo "Multi-GPU SIREN Training"
echo "========================================="
echo "Number of GPUs: ${NUM_GPUS}"
echo "Master Address: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "${PYTHONPATH}"
echo "========================================="

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --list-gpus

# Run training
echo "Starting distributed training..."

# Using torch.distributed.launch (compatible with PyTorch 1.9.0)
python3 -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_siren_hdf5_data_v2.py \
    --config config_siren_hdf5_data.yaml \
    --wandb-project flashmatch-siren-hdf5-data

# Alternative: Using torchrun (for PyTorch >= 1.10, if available)
# torchrun \
#     --nproc_per_node=${NUM_GPUS} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     train_siren_hdf5_mccorsika_v2_multigpu.py \
#     --config config_siren_hdf5_mccorsika_multigpu.yaml \
#     --wandb-project flashmatch-siren-multigpu
