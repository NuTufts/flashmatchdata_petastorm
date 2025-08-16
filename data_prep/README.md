# FlashMatch Data Preparation Pipeline

This directory contains a complete workflow for preparing training data for training a neural-network model of the optical response of the MicroBooNE detector using real cosmic ray data. The goal is to predict scintillation light patterns (PMT photoelectron counts) from 3D spatial charge configurations.

## Overview

The neural network training objective is to predict the 32-element PMT photoelectron vector given a variable-length sequence of 3D voxel features representing energy deposited by charged partices in the detector. 

## Architecture

### Core Components

The data preparation pipeline consists of several interconnected C++ classes and Python utilities:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ROOT Input    │───▶│  C++ Pipeline   │───▶│  HDF5 Output    │
│  (Cosmic Reco)  │    │ (main.cxx)      │    │ (ML Training)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌──────┴───────┐
                       │              │
                   ┌───▼───┐      ┌───▼────┐
                   │  CRT  │      │ Flash  │
                   │Matcher│      │Matcher │
                   └───────┘      └────────┘
```

The input file is made by a cosmic reconstruction algorithm that is part of the LANTERN reconstruction workflow.
The code for the cosmic reconstruction is found [here](https://github.com/NuTufts/larflow/blob/dlgen2_larmatchhdf5_retrain/larflow/Reco/CosmicParticleReconstruction.cxx).

The cosmic reconstruction can be run using the script found in `scripts/run_cosmic_reconstruction.sh`.

### Directory Structure

```
data_prep/
├── src/                           # C++ source implementation
│   ├── main.cxx                   # Main processing pipeline
│   ├── CosmicRecoInput.cxx        # Reads cosmic reco. ROOT file
│   ├── FlashTrackMatcher.cxx      # Flash-track association algorithms
│   ├── CRTMatcher.cxx             # CRT hit/track matching to cosmic track
│   ├── FlashMatchHDF5Output.cxx   # HDF5 data export
│   ├── FlashMatchOutputData.cxx   # ROOT data export
│   ├── LarliteDataInterface.cxx   # Data structure conversion
│   └── PrepareVoxelOutput.cxx     # Voxelization for ML input
├── include/                       # C++ header files
│   ├── DataStructures.h           # Core data structure definitions
│   ├── CosmicRecoInput.h          # Data loading interface
│   ├── FlashTrackMatcher.h        # Flash matching interface
│   ├── CRTMatcher.h               # CRT matching interface
│   ├── FlashMatchHDF5Output.h     # HDF5 output interface
│   ├── FlashMatchOutputData.h     # ROOT output interface
│   ├── LarliteDataInterface.h     # Data conversion utilities
│   └── PrepareVoxelOutput.h       # Voxelization interface
├── python/                        # Python utilities and training
│   ├── read_flashmatch_hdf5.py    # HDF5 data loading for PyTorch
│   ├── flashmatch_mixup.py        # MixUp data augmentation
│   └── train_with_mixup_example.py # Example training script
│   └── debug_shapes.py           # Checks for incorrect shapes
├── visualizations/               # Interactive data visualization
│   ├── cosmic_dashboard.py       # Cosmic reco input data dashboard
│   ├── match_dashboard.py        # Root data output dashboard
│   ├── hdf5_match_dashboard.py   # HDF5 data output dashboard
│   └── compare_match_flash_predictions.py # Prediction comparison
├── studies/                      # Data analysis utilities
│   └── calculate_means_vars.py   # Dataset statistics calculation
├── tuftscluster/                 # Scripts to run over data on Tufts
│   └── run_gridjob_hdf5_dataprep.sh    # runs code on worker node
│   └── submit_run_gridjob_dataprep.sh  # manages grid submission
└── build/                        # CMake build directory
```

## Core C++ Classes

### 1. Data Loading (`CosmicRecoInput`)
- **Purpose**: Loads cosmic ray reconstruction data from ROOT files
- **Key Methods**: 
  - `load_entry(int)`: Load specific event
  - `get_track_v()`: Access cosmic tracks
  - `get_opflash_v()`: Access optical flashes
  - `get_crttrack_v()`, `get_crthit_v()`: Access CRT data

### 2. Flash-Track Matching (`FlashTrackMatcher`)
- **Purpose**: Associates optical flashes with cosmic ray tracks
- **Algorithms**:
  - **Anode Crossing Matching**: Primary method using drift time
  - **Cathode Crossing Matching**: Alternative for cathode-crossing tracks
- **Key Methods**:
  - `FindAnodeCathodeMatches()`: Main matching algorithm
  - `LoadConfigFromFile()`: Load matching parameters

### 3. CRT Matching (`CRTMatcher`)
- **Purpose**: Correlates CRT hits/tracks with cosmic rays and optical flashes
- **Key Methods**:
  - `FilterCRTTracksByFlashMatches()`: CRT-flash correlation
  - `FilterCRTHitsByFlashMatches()`: CRT hit-flash correlation  
  - `MatchToCRTTrack()`: Match cosmic tracks to CRT tracks
  - `MatchToCRTHits()`: Match cosmic tracks to CRT hits

### 4. Voxelization (`PrepareVoxelOutput`)
- **Purpose**: Convert 3D tracks into voxelized representation for ML
- **Key Methods**:
  - `makeVoxelChargeTensor()`: Create voxel tensors from track data

### 5. HDF5 Output (`FlashMatchHDF5Output`)
- **Purpose**: Export data in HDF5 format for ML training
- **Key Methods**:
  - `storeEventVoxelData()`: Store complete event data
  - Batched writing for efficiency

## HDF5 Data Schema

The output HDF5 files are structured for efficient machine learning data loading:

### File Organization
```
flashmatch_data.h5
├── /voxel_data/           # Main data group
│   ├── entry_0/           # Individual match entries
│   │   ├── planecharge    # (N, 3) float32 - charge per wire plane
│   │   ├── indices        # (N, 3) int32   - voxel grid indices  
│   │   ├── avepos         # (N, 3) float32 - average 3D positions [cm]
│   │   ├── centers        # (N, 3) float32 - voxel center positions [cm]
│   │   ├── observed_pe_per_pmt     # (32,) float32 - observed PMT PE
│   │   ├── predicted_pe_per_pmt    # (32,) float32 - predicted PMT PE  
│   │   ├── observed_total_pe       # scalar float32 - total observed PE
│   │   ├── predicted_total_pe      # scalar float32 - total predicted PE
│   │   ├── match_type             # scalar int32 - type of match made
│   │   └── [event metadata]       # run, subrun, event, match_index
│   ├── entry_1/
│   └── ...
└── /event_info/          # Event-level metadata
```

### Data Types and Dimensions

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `planecharge` | (N, 3) | float32 | Charge deposition on U, V, Y wire planes |
| `indices` | (N, 3) | int32 | Voxel grid indices (ix, iy, iz) |
| `avepos` | (N, 3) | float32 | Average 3D position of charge [cm] |
| `centers` | (N, 3) | float32 | Voxel center coordinates [cm] |
| `observed_pe_per_pmt` | (32,) | float32 | Measured photoelectrons per PMT |
| `predicted_pe_per_pmt` | (32,) | float32 | Predicted photoelectrons per PMT using current MicroBooNE lightmodel |
| `observed_total_pe` | scalar | float32 | Total observed photoelectrons |
| `predicted_total_pe` | scalar | float32 | Total predicted photoelectrons |
| `match_type` | scalar | int32 | Matching algorithm used |

For the match_type, the value corresponds to:

| match_type value | Description |
| ---------------- | ----------- |
| -1 | Unassigned |
| 0  | Anode-crossing match |
| 1  | Cathode-crossing match |
| 2  | CRT Track matched to TPC track |
| 3  | CRT Hit matched to TPC track |
| 4  | Track-to-flash match based on time-consistency (not used, still needs devlopment) |

**Note**: N varies per entry (typically 50-500 voxels per cosmic track)

## Python Data Loading

### PyTorch Integration

The `read_flashmatch_hdf5.py` module provides PyTorch-compatible data loading:

```python
from read_flashmatch_hdf5 import FlashMatchVoxelDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = FlashMatchVoxelDataset(
    hdf5_files='filelist.txt',
    max_voxels=500,
    load_to_memory=False
)

# Create data loader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
for batch in dataloader:
    features = batch['features']  # (batch, max_voxels, 6)
    targets = batch['observed_pe_per_pmt']  # (batch, 32)
    mask = batch['mask']  # (batch, max_voxels)
    # ... training code
```

### MixUp Data Augmentation

MixUp augmentation is available for improved training:

```python
from flashmatch_mixup import create_mixup_dataloader

# MixUp combines two samples:
# - PMT targets: α * pe_a + (1-α) * pe_b  
# - Voxel features: concatenate [voxels_a * α, voxels_b * (1-α)]

mixup_loader = create_mixup_dataloader(
    base_dataset,
    batch_size=32,
    mixup_prob=0.5,    # 50% chance of applying mixup
    alpha=1.0,         # Beta distribution parameter
    max_total_voxels=1000
)
```

## Build Instructions

### Environment Setup

**Required**: MicroBooNE container environment with ubdl stack environment setup:

```bash
# On Tufts cluster
module load apptainer/1.2.4-suid
singularity shell --bind /cluster/tufts/wongjiradlabnu:/cluster/tufts/wongjiradlabnu \
    /cluster/tufts/wongjiradlabnu/larbys/larbys-container/u20.04_cu111_cudnn8_torch1.9.0_minkowski_npm.sif

# Setup ubdl environment
source setenv_py3_container.sh
source configure_container.sh
```

Note that the ubdl repo must be built (see ubdl repo for info).

### Compilation

```bash
# Create build directory
mkdir -p build && cd build

# Configure (Release mode)
cmake ..

# Configure (Debug mode)  
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build
make -j4

# Install to build/installed
make install
```

## Usage Examples

### Basic Data Processing

```bash
# Process cosmic ray data with flash/CRT matching
./build/installed/bin/main \
    --input cosmic_reco_input.root \
    --output-hdf5 flashmatch_output.h5 \
    --larcv larcv_input.root \
    --max-events 1000

# Optional: Also save ROOT format
./build/installed/bin/main \
    --input cosmic_reco_input.root \
    --output-root flashmatch_output.root \
    --output-hdf5 flashmatch_output.h5 \
    --larcv larcv_input.root
```

Note that in the above example, `cosmic_reco_input.root` and `larcv_input.root` will have been made by the cosmic reconstruction program.

### Python Data Analysis

```bash
# Calculate dataset statistics for normalization
python studies/calculate_means_vars.py \
    -i filelist.txt \
    -o data_statistics.root \
    --max-entries 10000

# Test data loading performance
python test_hdf5_dataloader.py filelist.txt

# Train with MixUp augmentation (still need to test)
python train_with_mixup_example.py \
    --filelist filelist.txt \
    --batch-size 32 \
    --mixup-prob 0.5 \
    --epochs 50
```

## Visualization and Debugging

### Interactive Dashboards

The visualization directory provides several Dash-based web interfaces:

#### 1. ROOT Data Dashboard (`cosmic_dashboard.py`)
```bash
python visualizations/cosmic_dashboard.py cosmic_reco_data.root
# Open browser to http://localhost:8050
```
- **Features**: 3D track visualization, PMT displays, timing correlations
- **Use case**: Inspect raw cosmic ray reconstruction data

#### 2. Match Results Dashboard (`match_dashboard.py`) 
```bash
python visualizations/match_dashboard.py flashmatch_output.root
# Open browser to http://localhost:8050
```
- **Features**: Flash-track match visualization, CRT correlations
- **Use case**: Validate matching algorithm performance

#### 3. HDF5 Data Dashboard (`hdf5_match_dashboard.py`)
```bash  
python visualizations/hdf5_match_dashboard.py flashmatch_output.h5
# Open browser to http://localhost:8050
```
- **Features**: ML training data visualization, voxel displays
- **Use case**: Inspect final training datasets

#### 4. Prediction Comparison (`compare_match_flash_predictions.py`)
```bash
python visualizations/compare_match_flash_predictions.py \
    flashmatch_output.root --entry 42
```
- **Features**: Compare observed vs predicted PMT patterns
- **Use case**: Validate flash prediction accuracy

### Dashboard Features

All dashboards provide:
- **3D Track Visualization**: Interactive 3D plots of cosmic ray tracks
- **PMT Pattern Display**: Observed and predicted photoelectron patterns
- **Timing Analysis**: Flash timing vs track crossing times
- **CRT Correlation**: CRT hit/track associations
- **Quality Metrics**: Match quality and filtering statistics
- **Entry Navigation**: Browse through events/matches

### Debug Mode

Enable verbose logging:
```bash
export FLASHMATCH_DEBUG=1
export FLASHMATCH_LOG_LEVEL=DEBUG

./build/installed/bin/main --input data.root --output out.h5 --debug
```

## Configuration and Customization

### Key Parameters

The main processing pipeline can be customized through command-line arguments:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max-events N` | Limit number of events processed | All events |
| `--start-event N` | Starting event index | 0 |
| `--verbosity N` | Logging level (0-3) | 1 |
| `--no-crt` | Disable CRT matching | CRT enabled |
| `--debug` | Enable debug output | Disabled |

### Data Processing Filters

The pipeline applies several data quality filters:
- **Flash PE threshold**: Minimum 1.0 total photoelectrons
- **Flash prediction agreement**: log(predicted/observed) PE within [-2, 2]
- **Voxel count limits**: Maximum voxels per track (configurable)

## Development and Extension

### Adding New Matching Algorithms

To implement new flash-track matching methods:

1. **Add method to `FlashTrackMatcher` class**:
```cpp
int FlashTrackMatcher::YourNewMatchingMethod(
    const EventData& input_data,
    EventData& output_data
);
```

2. **Call from main processing loop** in `main.cxx`:
```cpp
int new_matches = flash_matcher.YourNewMatchingMethod(input_data, output_data);
```

3. **Update configuration** as needed

### Extending HDF5 Schema

To add new data fields to the HDF5 output:

1. **Update `FlashMatchHDF5Output.h`** with new dataset names
2. **Modify `storeEventVoxelData()`** to write new fields
3. **Update Python data loading** in `read_flashmatch_hdf5.py`

## Troubleshooting

### Common Build Issues

**Missing ubdl environment**:
```bash
source /path/to/ubdl/setenv_py3_container.sh
source /path/to/ubdl/configure_container.sh
```

**CMake configuration errors**:
```bash
export CMAKE_PREFIX_PATH=$LARLITE_LIBDIR:$LARCV_LIBDIR:$CMAKE_PREFIX_PATH
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## Future Development/TODO list

* Provide a module to extend track through non-TPC areas (need liquid volume model -- or just check where light model is non-zero?).  Need a scheme to assign charge in these extension voxels. Maybe use the average charge in the other voxels in the track.
* Clean-up repeated checks for tracks being at image bounds
* Reduce mem usage by passing single copy of SpaceChargeMicroBooNE to the different utilities.
* Refactor repeated sce application into one location instead of repeated in the different matching algorithms.
* Put geometry info into a text file and load into a class that is passed to the algorithms. Also provide option to extract this info from the larlite interfaces.
* Modify data loader to give warning when empty entry found -- probably is corrupt file and so should remove it from the data sample?