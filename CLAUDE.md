# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashMatchNet is a deep learning project for matching optical flash data with particle tracks in liquid argon time projection chambers (LArTPC) for neutrino physics experiments. It uses Petastorm for efficient data storage and PyTorch/MinkowskiEngine for neural network training.

## Common Development Commands

### Environment Setup
```bash
# Set up Python paths for dependencies
source setenv.sh
```

### Training Models

Submit training jobs via SLURM:
```bash
# Submit MLP training job
sbatch submit_train_mlp_p1cmp075.sh

# Or run training directly (on appropriate hardware)
python train_mlp.py
python train_siren.py
python train_lightmodel.py
```

### Data Preparation

Create training data from ROOT files:

**Petastorm format (legacy):**
```bash
python make_flashmatch_training_data.py \
  -db /path/to/output/database/ \
  -lcv /path/to/larcv/file \
  -mc /path/to/mcinfo/file \
  -op /path/to/opreco/file \
  --port 5000 \
  --over-write
```

**HDF5 format (recommended):**
```bash
python flashmatch_hdf5_writer.py \
  -o /path/to/output.h5 \
  -lcv /path/to/larcv/file \
  -mc /path/to/mcinfo/file \
  -op /path/to/opreco/file \
  -n 1000
```

### Running Tests
```bash
# Test flashmatch code
python test_flashmatch_code.py

# Test PyTorch data reader
python pytorch_reader_test.py

# Test HDF5 reader/writer
python example_hdf5_usage.py --all
```

### Model Inference and Analysis
```bash
# Run model inference analysis
python model_inference_analysis.py

# Data studies
python data_studies.py

# Visualize flash match data (Jupyter notebook)
jupyter notebook view_flashmatch_data.ipynb
```

## High-Level Architecture

### Core Components

1. **Data Pipeline (`flashmatchnet/data/`)**
   - `petastormschema.py`: Defines the FlashMatchSchema for storing event data, 3D coordinates, and PMT signals
   - `flashmatchdata.py`: Data loader utilities for PyTorch
   - Handles sparse 3D point clouds with features and 32 PMT photoelectron readings

2. **Models (`flashmatchnet/model/`)**
   - Multiple architectures: MLP, SIREN, ResNet-based models
   - Supports both dense and sparse (MinkowskiEngine) implementations
   - Models map 3D particle tracks to expected PMT signals

3. **Loss Functions (`flashmatchnet/losses/`)**
   - Poisson negative log-likelihood with Earth Mover's Distance (EMD)
   - Custom geometric losses via the geomloss submodule
   - Designed for comparing predicted vs actual PMT light patterns

4. **External Dependencies (`dependencies/`)**
   - `geomloss`: Optimal transport and geometric losses
   - `siren-pytorch`: SIREN (Sinusoidal Representation Networks) implementation
   - Managed as git submodules

### Data Flow

#### Legacy Petastorm Pipeline
1. **Input**: ROOT files containing LArTPC simulation/reconstruction data
2. **Processing**: `make_flashmatch_training_data.py` converts ROOT → Petastorm/Parquet
3. **Training**: PyTorch DataLoaders read Petastorm data for model training
4. **Output**: Trained models predict PMT light patterns from 3D particle tracks

#### New HDF5 Pipeline (Recommended)
1. **Input**: ROOT files containing LArTPC simulation/reconstruction data
2. **Processing**: `flashmatch_hdf5_writer.py` converts ROOT → HDF5
3. **Training**: `flashmatch_hdf5_reader.py` provides PyTorch DataLoaders for HDF5 data
4. **Output**: Trained models predict PMT light patterns from 3D particle tracks

### Key Technologies

- **Storage**: Petastorm (Parquet-based) for efficient large-scale data handling
- **Compute**: SLURM job scheduling, Singularity containers
- **ML Stack**: PyTorch, MinkowskiEngine (sparse 3D convolutions), Weights & Biases
- **Physics Tools**: ROOT, larlite, larcv for neutrino detector data

### Development Notes

- The project uses HPC resources with GPU requirements (typically P100 GPUs)
- Training jobs are long-running (up to 6 days) and checkpoint frequently
- Data is stored in distributed Petastorm databases accessed via file:// URLs
- The `NPMTS=32` constant appears throughout - this is the number of photomultiplier tubes
- Models are trained to minimize the difference between predicted and actual light patterns