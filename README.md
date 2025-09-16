# FlashMatch Neural Network Project

This project implements deep learning models to match optical flashes with charged particle tracks in liquid argon time projection chambers (LArTPC) for neutrino physics experiments. The core challenge is predicting the expected light pattern (photoelectron counts on photomultiplier tubes) from 3D particle trajectories.

## Table of Contents
- [Project Overview](#project-overview)
- [Code Organization](#code-organization)
- [Training Data](#training-data)
- [Model Training](#model-training)
- [Inference and Analysis](#inference-and-analysis)
- [Getting Started](#getting-started)
- [Key Files Reference](#key-files-reference)

## Project Overview

### The Physics Problem
In liquid argon detectors, charged particles create two types of signals:
1. **Ionization tracks**: 3D trajectories reconstructed from wire chamber data
2. **Scintillation light**: Optical flashes detected by photomultiplier tubes (PMTs)

The goal is to train neural networks that can predict the expected PMT light pattern given a 3D particle track, enabling better event reconstruction and particle identification.

### Technical Approach
- **Input**: 3D voxelized particle tracks with charge information
- **Output**: Predicted photoelectron (PE) counts for 32 PMTs
- **Models**: MLP, SIREN, and sparse convolutional architectures
- **Loss Functions**: Poisson negative log-likelihood with Earth Mover's Distance (EMD)

## Code Organization

```
flashmatchdata_petastorm/
├── flashmatchnet/              # Main Python package
│   ├── data/                   # Data loading and preprocessing
|   |   ├── read_flashmatch_hdf5.py # Read the current training data
|   |   ├── flashmatch_mixup.py # Load the training data. Mix two examples together when loading.
│   │   ├── petastormschema.py  # (deprecated) Legacy Petastorm data schema
│   │   ├── reader.py           # (deprecated) Legacy Petastorm data reader
│   │   ├── flashmatchdata.py   # (deprecated) Legacy data utilities
|   |   └── flashmatch_hdf5_reader.py # (deprecated) Legacy PyTorch DataLoader for HDF5
│   ├── model/                  # Neural network architectures
│   │   ├── flashmatchMLP.py    # Multi-layer perceptron model
│   │   └── ...                 # Other model architectures
│   ├── losses/                 # Loss function implementations
│   │   └── loss_poisson_emd.py # Main loss function
│   └── utils/                  # Utility functions
│       ├── pmtutils.py         # PMT geometry utilities
│       └── coord_and_embed_functions.py # Coordinate embeddings
├── dependencies/               # Git submodules
│   ├── geomloss/              # Optimal transport losses
│   └── siren-pytorch/         # SIREN neural networks
├── data_prep/                 # Code and scripts to prepare training data
├── mcstudy_prep/              # (deprecated) Corsika-simulation MC Data preparation scripts 
├── analysis/                  # Analysis and visualization tools (TODO)
│
# New HDF5 Data System (Recommended)
├── arxiv/flashmatch_hdf5_writer.py  # Convert ROOT → HDF5
├── arxiv/train_mlp_hdf5.py          # Training script using HDF5
├── arxiv/example_hdf5_usage.py      # Example usage and testing
│
# Legacy Petastorm System (deprecated)
├── arvix/make_flashmatch_training_data.py  # Convert ROOT → Petastorm
├── arvix/train_mlp.py               # MLP training (Petastorm)
├── arxiv/train_siren.py             # SIREN training (Petastorm). Used on "v3" data. Trained relatively well.
├── arxiv/train_lightmodel.py        # Light model training
│
# Analysis and Inference (deprecated)
├── arxiv/model_inference_analysis.py # Run inference on trained models
├── arxiv/data_studies.py            # Data exploration scripts
├── arxiv/view_flashmatch_data.ipynb # Jupyter notebook for visualization
│
# Job Submission (HPC)
├── submit_train_mlp_p1cmp075.sh # SLURM job submission script
└── setenv.sh                  # Environment setup
```

## Training Data

### What is the Training Data?

The training data consists of **matched pairs** of:
1. **3D Voxelized Tracks**: Particle trajectories discretized into 5cm³ voxels
   - **Coordinates**: (x,y,z) indices in the voxel grid
   - **Features**: Charge deposition per wire plane (3 values per voxel)
   - **Truth Labels**: Particle ancestor ID for physics interpretation

2. **PMT Flash Data**: Measured light signals
   - **Flash PE**: Photoelectron counts for each of 32 PMTs
   - **Timing**: Flash time matched to particle crossing time
   - **Quality**: Filtered for reasonable PE thresholds

### Data Schema

Each training example contains:
```python
{
    'sourcefile': str,     # Source ROOT filename
    'run': int32,          # Run number
    'subrun': int32,       # Subrun number  
    'event': int32,        # Event number
    'matchindex': int32,   # Flash index within event
    'ancestorid': int32,   # Particle ancestor ID
    'coord': int64[N,3],   # Voxel coordinates (N voxels)
    'feat': float32[N,3],  # Charge features per plane
    'flashpe': float32[1,32] # PE counts for 32 PMTs
}
```

### How Training Data is Created

#### Option 1: HDF5 Pipeline (Recommended)

**Script**: `flashmatch_hdf5_writer.py`

```bash
python flashmatch_hdf5_writer.py \
  -o output_data.h5 \
  -lcv /path/to/larcv_truth.root \
  -mc /path/to/mcinfo.root \
  -op /path/to/opreco.root \
  -n 1000  # number of events to process
```

**Process**:
1. **Input Files**:
   - `larcv_truth.root`: Wire plane images and truth particle information
   - `mcinfo.root`: Monte Carlo truth information
   - `opreco.root`: Reconstructed optical flashes

2. **Voxelization**: Uses `VoxelizeTriplets` class to:
   - Convert 2D wire images to 3D spacepoints
   - Apply truth labels from simulation
   - Group spacepoints into 5cm voxels
   - Correct for drift time using truth information

3. **Flash Matching**: Uses `OpModelMCDataPrep` utility to:
   - Match reconstructed flashes to true particle information
   - Filter good quality matches
   - Extract PMT PE values

4. **Output**: Single HDF5 file with variable-length arrays

#### Option 2: Petastorm Pipeline (Legacy)

**Script**: `make_flashmatch_training_data.py`

```bash
python make_flashmatch_training_data.py \
  -db /path/to/petastorm/database/ \
  -lcv /path/to/larcv_truth.root \
  -mc /path/to/mcinfo.root \
  -op /path/to/opreco.root \
  --port 5000
```

Creates distributed Parquet files via PySpark (more complex infrastructure).

### Data Preprocessing

Key preprocessing steps in both pipelines:
- **TPC Boundary Filtering**: Remove voxels outside detector volume
- **Coordinate Normalization**: Subtract TPC origin for relative coordinates  
- **Charge Normalization**: Scale charge values to reasonable ranges
- **PE Normalization**: Scale photoelectron counts for neural network training

## Model Training

### Available Models

1. **FlashMatchMLP** (`flashmatchnet/model/flashmatchMLP.py`)
   - Multi-layer perceptron with coordinate embeddings
   - Input: 112 features (coordinates + embeddings + charge)
   - Output: 32 PMT predictions

2. **SIREN Models** (`train_siren.py`)
   - Sinusoidal representation networks
   - Good for continuous coordinate spaces

3. **Sparse Convolutional Models**
   - Use MinkowskiEngine for efficient 3D convolutions
   - Handle variable-size voxel inputs
   - Still a TODO

### Training Scripts

#### HDF5 Training (Recommended)
```bash
# Edit file paths in train_mlp_hdf5.py first
python train_mlp_hdf5.py
```

#### Petastorm Training (Legacy)
```bash
python train_mlp.py      # MLP model
python train_siren.py    # SIREN model
```

### Key Training Components

1. **Loss Function** (`flashmatchnet/losses/loss_poisson_emd.py`):
   ```python
   PoissonNLLwithEMDLoss(magloss_weight=1.0,
                         mag_loss_on_sum=False, 
                         full_poisson_calc=False)
   ```
   - Poisson negative log-likelihood for total PE predictions
   - Earth Mover's Distance for spatial pattern matching

2. **Data Loading**:
   - Batch size: typically 32
   - Workers: 4 parallel data loading processes
   - Shuffle: enabled for training

3. **Optimization**:
   - AdamW optimizer
   - Learning rates: 1e-5 (general), 1e-7 (light yield parameter)
   - Checkpointing every 1000 iterations

### HPC Job Submission

For long training runs on compute clusters:
```bash
# Edit paths in submit script first
sbatch submit_train_mlp_p1cmp075.sh
```

Uses Singularity containers with pre-installed dependencies.

## Inference and Analysis

### Running Inference

**Main Script**: `model_inference_analysis.py`

This script loads trained models and runs inference on test data to evaluate performance.

**Key Functions**:
1. **Model Loading**: Load checkpoint files with trained weights
2. **Data Processing**: Prepare test examples in same format as training
3. **Prediction**: Run forward pass through trained network
4. **Metrics Calculation**: Compare predictions to ground truth

### Analysis Tools

1. **Data Exploration** (`data_studies.py`):
   - Analyze training data distributions
   - Visualize voxel patterns and PMT responses
   - Quality control checks

2. **Interactive Analysis** (`view_flashmatch_data.ipynb`):
   - Jupyter notebook for detailed data inspection
   - 3D visualization of particle tracks
   - PMT pattern analysis

3. **Validation Metrics** (`flashmatchnet/utils/trackingmetrics.py`):
   - Physics-motivated performance metrics
   - Flash-matching efficiency calculations
   - Spatial resolution measurements

### Key Analysis Outputs

- **Prediction vs Truth Plots**: Compare predicted and actual PE patterns
- **Residual Analysis**: Study systematic biases in predictions  
- **Efficiency Curves**: Flash-matching performance vs various cuts
- **Physics Validation**: Verify model makes physical sense

### Performance Metrics

Common metrics for evaluating models:
- **Poisson NLL**: Primary loss function value
- **Mean Absolute Error**: Simple PE prediction accuracy
- **Earth Mover's Distance**: Spatial pattern similarity
- **Flash Matching Efficiency**: Physics-level performance

## Getting Started

### Prerequisites
```bash
# Set up environment (adds dependencies to Python path)
source setenv.sh

# Required packages:
# - PyTorch, MinkowskiEngine
# - h5py (for HDF5 data)
# - ROOT, larcv, larlite (for physics libraries)
# - wandb (for experiment tracking)
```

### Quick Start

1. **Test the HDF5 system**:
   ```bash
   python example_hdf5_usage.py --all
   ```

2. **Create training data** (if you have ROOT files):
   ```bash
   python flashmatch_hdf5_writer.py \
     -o test_data.h5 \
     -lcv your_larcv_file.root \
     -mc your_mcinfo_file.root \
     -op your_opreco_file.root \
     -n 100  # small test dataset
   ```

3. **Train a model** (after editing file paths):
   ```bash
   python train_mlp_hdf5.py
   ```

4. **Monitor training**:
   - Check terminal output for loss values
   - View Weights & Biases dashboard if enabled
   - Check checkpoint files in checkpoint directory

### Development Workflow

1. **Data Preparation**: Create HDF5 training data from ROOT files
2. **Model Development**: Modify architectures in `flashmatchnet/model/`
3. **Training**: Run training scripts with different hyperparameters
4. **Evaluation**: Use analysis scripts to assess model performance
5. **Iteration**: Refine based on physics validation

## Key Files Reference

### Most Important Files for New Developers

| File | Purpose | When to Modify |
|------|---------|----------------|
| `flashmatch_hdf5_writer.py` | Create training data | Change data processing logic |
| `flashmatchnet/data/flashmatch_hdf5_reader.py` | Load training data | Modify data augmentation |
| `train_mlp_hdf5.py` | Train MLP model | Adjust training parameters |
| `flashmatchnet/model/flashmatchMLP.py` | MLP architecture | Change model design |
| `flashmatchnet/losses/loss_poisson_emd.py` | Loss function | Modify training objective |
| `model_inference_analysis.py` | Run inference | Evaluate trained models |
| `example_hdf5_usage.py` | Test/debug system | Understanding data format |

### Configuration Files

- File paths and hyperparameters are typically hardcoded in training scripts
- Edit the constants at the top of training scripts to change:
  - Data file locations
  - Batch size, learning rate
  - Checkpoint directories
  - Weights & Biases settings

### Common Issues

1. **File Paths**: Update hardcoded paths in training scripts
2. **Dependencies**: Ensure physics libraries (ROOT, larcv, larlite) are available
3. **Memory**: Large datasets may require adjustment of batch size or workers
4. **CUDA**: Verify GPU availability for training

This codebase represents a complete pipeline from raw detector simulation data to trained neural networks for neutrino physics applications. The HDF5 system provides a more maintainable and efficient approach compared to the legacy Petastorm infrastructure.
