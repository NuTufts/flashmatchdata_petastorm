# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashMatchNet is a deep learning project for matching optical flash data with particle tracks in liquid argon time projection chambers (LArTPC) for neutrino physics experiments. It uses PyTorch and SIREN networks to predict photomultiplier tube (PMT) light patterns from 3D voxelized particle tracks.

## Common Development Commands

### Environment Setup
```bash
# Set up Python paths for dependencies
source setenv.sh

# On Tufts cluster with container
module load apptainer/1.2.4-suid
singularity shell --bind /cluster/tufts/wongjiradlabnu:/cluster/tufts/wongjiradlabnu \
    /cluster/tufts/wongjiradlabnu/larbys/larbys-container/u20.04_cu111_cudnn8_torch1.9.0_minkowski_npm.sif
source setenv_py3_container.sh
source configure_container.sh
```

### Training Models

**SIREN model with HDF5 data (recommended):**
```bash
# Run training directly
python3 train_siren_hdf5_data_v2.py

# Or submit via SLURM
sbatch submit_train_siren_hdf5_data_v2.sh
```

### Data Preparation

**Build C++ data preparation tools:**
```bash
cd data_prep
mkdir -p build && cd build
cmake ..
make -j4
make install
```

**Create HDF5 training data from cosmic reconstruction:**
```bash
./data_prep/build/installed/bin/main \
  --input cosmic_reco_input.root \
  --output-hdf5 flashmatch_output.h5 \
  --larcv larcv_input.root \
  --max-events 1000
```

**Calculate dataset statistics:**
```bash
python data_prep/studies/calculate_means_vars.py \
  -i filelist.txt \
  -o data_statistics.root \
  --max-entries 10000
```

### Running Tests
```bash
# Test HDF5 data system
python arxiv/example_hdf5_usage.py --all

# Debug data shapes
python data_prep/debug_shapes.py

# Test data loader
python data_prep/debug_dataloader.py
```

## High-Level Architecture

### Core Components

1. **Data Pipeline (`flashmatchnet/data/` and `data_prep/`)**
   - `read_flashmatch_hdf5.py`: PyTorch dataset for loading HDF5 training data
   - `flashmatch_mixup.py`: MixUp data augmentation for improved training
   - C++ pipeline in `data_prep/` converts cosmic ray reconstruction ROOT files to HDF5

2. **Models (`flashmatchnet/model/`)**
   - `lightmodel_siren.py`: SIREN-based light model (primary architecture)
   - `flashmatchMLP.py`: MLP model with coordinate embeddings
   - Models map 3D voxelized tracks to 32 PMT photoelectron predictions

3. **Loss Functions (`flashmatchnet/losses/`)**
   - `loss_poisson_emd.py`: Combined Poisson NLL + Earth Mover's Distance loss
   - Designed for comparing predicted vs actual PMT light patterns

4. **External Dependencies (`dependencies/`)**
   - `geomloss`: Optimal transport and geometric losses
   - `siren-pytorch`: SIREN (Sinusoidal Representation Networks) implementation

### Data Flow

1. **Input**: Cosmic ray reconstruction ROOT files from MicroBooNE detector
2. **C++ Processing**: `data_prep/src/main.cxx` performs:
   - Flash-track matching using drift time
   - CRT correlation for timing validation
   - Voxelization into 5cm³ grid
3. **HDF5 Output**: Variable-length arrays with:
   - Voxel coordinates and charge features
   - Observed PMT photoelectron counts (32 values)
   - Predicted PMT values from current light model
4. **Training**: PyTorch loads HDF5 data for SIREN model training
5. **Output**: Trained models predict PMT patterns from particle tracks

### HDF5 Data Schema

Each training example contains:
- `planecharge`: (N, 3) float32 - Charge per wire plane (U, V, Y)
- `indices`: (N, 3) int32 - Voxel grid indices
- `avepos`/`centers`: (N, 3) float32 - 3D positions [cm]
- `observed_pe_per_pmt`: (32,) float32 - Measured PMT signals
- `predicted_pe_per_pmt`: (32,) float32 - Current model predictions
- `match_type`: int32 - Matching algorithm used (0=anode, 1=cathode)

### Key Configuration Files

- `config_siren_hdf5_data.yaml`: Main training configuration
- `train_no_anode_mcc9_v29e_dl_run3_G1_extbnb_dlana.txt`: Training file list
- `valid_no_anode_mcc9_v29e_dl_run3_G1_extbnb_dlana.txt`: Validation file list

### Key Technologies

- **ML Stack**: PyTorch, SIREN networks, Weights & Biases for tracking
- **Data Format**: HDF5 for efficient storage and loading
- **Physics Tools**: ROOT, larlite, larcv for detector data
- **Compute**: SLURM scheduling on Tufts cluster with GPU nodes
- **Constants**: NPMTS=32 (number of photomultiplier tubes in MicroBooNE)