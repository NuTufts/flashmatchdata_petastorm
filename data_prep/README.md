# Training Data Preparation for Flash-Track Matching with Real Data

This directory contains the complete workflow for preparing training data using real cosmic ray data from MicroBooNE detector runs. The goal is to create a dataset for training neural networks to predict scintillation light patterns from spatial charge configurations.

## Project Overview

The neural network training objective is to predict the amount of scintillation light observed at photomultiplier tubes (PMTs) given a 3D spatial configuration of charge deposition from cosmic ray tracks. This real-data approach provides ground truth light measurements while controlling for systematic uncertainties present in simulation.

## Workflow Overview

The data preparation pipeline consists of four main steps:

1. **Cosmic Ray Reconstruction**: Extract cosmic ray tracks and correlate with optical flashes and CRT information
2. **Quality Cuts**: Apply selection criteria to ensure clean, well-reconstructed events
3. **Flash-Track Matching**: Associate optical flashes with cosmic ray tracks using timing and geometry
4. **Data Format Conversion**: Convert to HDF5 format optimized for neural network training

## Directory Structure

```
data_prep/
├── CMakeLists.txt              # Main build configuration
├── README.md                   # This file
├── scripts/                    # Shell scripts for running workflow steps
│   ├── run_cosmic_reconstruction.sh
│   ├── submit_batch_processing.sh
│   └── setup_environment.sh
├── src/                        # C++ source code
│   ├── main.cxx               # Main program for steps 2-3
│   ├── CosmicTrackSelector.cxx # Quality cut implementation
│   ├── FlashTrackMatcher.cxx  # Flash-track matching algorithms
│   ├── CRTMatcher.cxx         # CRT hit/track matching
│   └── CMakeLists.txt         # Source build configuration
├── include/                    # C++ header files
│   ├── CosmicTrackSelector.h
│   ├── FlashTrackMatcher.h
│   ├── CRTMatcher.h
│   └── DataStructures.h
├── python/                     # Python utilities
│   ├── hdf5_converter.py      # Step 4: ROOT to HDF5 conversion
│   ├── data_validator.py      # Validate output data quality
│   └── config_parser.py       # Configuration file handling
├── visualizations/            # Visualization tools
│   ├── plot_tracks_flashes.py
│   ├── plot_flash_matches.py
│   └── display_crt_matches.py
├── build/                     # CMake build directory
└── config/                    # Configuration files
    ├── quality_cuts.yaml
    ├── flash_matching.yaml
    └── output_format.yaml
```

## Dependencies and Environment Setup

### Required Environment
You must work within the standard apptainer container environment. Load the container using:

```bash
# Navigate to ubdl directory first
cd /path/to/ubdl
source scripts/tufts_start_container.sh
```

### Required Software Stack
- **ubdl repository environment** (larlite, larcv, larflow, ublarcvapp)
- **ROOT** (for physics data I/O)
- **PyTorch** (for neural network integration)
- **HDF5** (for efficient training data storage)
- **OpenCV** (for geometric algorithms)
- **Eigen3** (optional, for linear algebra operations)

### Environment Setup
```bash
# Setup ubdl environment
source setenv_py3_container.sh
source configure_container.sh

# Setup data_prep specific environment
cd flashmatchdata_petastorm/data_prep
source scripts/setup_environment.sh
```

## Build Instructions

### Initial Build Setup
```bash
# Create and enter build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build the C++ programs
make -j4

# Install to build/installed
make install
```

### Incremental Builds
```bash
cd build
make -j4
```

## Detailed Workflow Steps

### Step 1: Cosmic Ray Reconstruction

**Purpose**: Run cosmic ray track reconstruction and save optical flash + CRT information.

**Implementation**: Uses `larflow::reco::CosmicParticleReconstruction` class to process:
- Input: dlmerged files (ADC images, ssnet, badch info)
- Output: Cosmic ray tracks with associated optical and CRT data

**Usage**:
```bash
# Single file processing
./scripts/run_cosmic_reconstruction.sh -i dlmerged_input.root -o cosmic_reco_output.root

# Batch processing for large datasets
./scripts/submit_batch_processing.sh --input-list dlmerged_files.txt --output-dir /path/to/output/
```

**Configuration**: Located in `config/cosmic_reconstruction.yaml`

### Step 2: Quality Cuts

**Purpose**: Apply selection criteria to ensure clean, well-reconstructed cosmic ray events.

**Implemented Quality Cuts**:
- **Boundary cuts**: Remove tracks near image boundaries (configurable distance thresholds)
- **Track quality**: Minimum track length, hit density requirements
- **Containment**: Require partial containment for energy calibration
- **Flash requirements**: Associated flash within timing window
- **CRT requirements**: CRT hit/track correlation (for Runs 3+)

**Configuration**: Modify `config/quality_cuts.yaml`:
```yaml
boundary_cuts:
  min_distance_to_edge: 10.0  # cm
  require_both_ends_contained: false

track_quality:
  min_track_length: 50.0      # cm
  min_hit_density: 0.5        # hits/cm
  max_gap_size: 5.0           # cm

flash_matching:
  timing_window: 23.4         # microseconds
  pe_threshold: 50.0          # minimum PE in flash

crt_matching:
  timing_tolerance: 1.0       # microseconds
  position_tolerance: 30.0    # cm
```

### Step 3: Flash-Track Matching

**Purpose**: Associate optical flashes with cosmic ray tracks using multiple criteria.

**Matching Algorithms**:
1. **Anode Crossing Time**: Match flash time to expected anode crossing
2. **Cathode Crossing Time**: Alternative timing for cathode-crossing tracks
3. **Degeneracy Resolution**: Handle multiple track-flash candidates
4. **CRT Track Matching**: Use CRT timing for additional constraints (Runs 3+)
5. **CRT Hit Matching**: Match individual CRT hits to track trajectory

**Configuration**: Modify `config/flash_matching.yaml`:
```yaml
timing_matching:
  anode_crossing_tolerance: 0.5    # microseconds
  cathode_crossing_tolerance: 0.5  # microseconds
  drift_velocity: 0.1098           # cm/microsecond

spatial_matching:
  track_flash_distance_cut: 100.0  # cm
  pmt_coverage_requirement: 0.3    # fraction

crt_integration:
  enable_crt_track_matching: true
  enable_crt_hit_matching: true
  crt_timing_precision: 1.0        # nanoseconds
```

### Step 4: HDF5 Conversion

**Purpose**: Convert matched flash-track pairs into HDF5 format optimized for neural network training.

**Data Schema Transformation**:
- **Input**: ROOT files with event-based organization
- **Output**: HDF5 files with flash-track match entries

**Schema Design**:
```python
# Each HDF5 entry contains:
entry = {
    'track_points': array(N, 3),        # 3D track coordinates [cm]
    'track_charge': array(N,),          # Charge deposition [ADC]
    'track_features': array(N, F),      # Additional track features
    'flash_pe': array(32,),             # PMT photoelectron counts
    'flash_time': float,                # Flash timing [microseconds]
    'crt_hits': array(M, 4),           # CRT hit positions + time
    'geometry_info': dict,              # Detector geometry metadata
    'quality_metrics': dict            # Quality scores for filtering
}
```

**Usage**:
```bash
python python/hdf5_converter.py \
    --input matched_data.root \
    --output training_data.h5 \
    --config config/output_format.yaml \
    --max-entries-per-file 10000
```

## Usage Examples

### Complete Pipeline Execution
```bash
# 1. Cosmic reconstruction
./scripts/run_cosmic_reconstruction.sh \
    --input dlmerged_run3_cosmic.root \
    --output cosmic_tracks_run3.root

# 2-3. Quality cuts and flash matching
./build/installed/bin/flashmatch_dataprep \
    --input cosmic_tracks_run3.root \
    --output matched_flashes_run3.root \
    --config config/quality_cuts.yaml \
    --config config/flash_matching.yaml

# 4. HDF5 conversion
python python/hdf5_converter.py \
    --input matched_flashes_run3.root \
    --output training_data_run3.h5 \
    --config config/output_format.yaml
```

### Batch Processing for Large Datasets
```bash
# Submit SLURM jobs for processing multiple files
./scripts/submit_batch_processing.sh \
    --input-list cosmic_files_run3.txt \
    --output-dir /data/flashmatch_training/ \
    --jobs-per-node 4 \
    --memory-per-job 8GB
```

## Data Validation and Quality Control

### Validation Tools
```bash
# Validate HDF5 output data
python python/data_validator.py \
    --input training_data_run3.h5 \
    --checks geometry,charge_conservation,timing

# Generate quality control plots
python visualizations/plot_flash_matches.py \
    --input matched_flashes_run3.root \
    --output plots/quality_control/
```

### Expected Output Statistics
- **Cosmic ray tracks**: ~1000-5000 per run
- **Quality-passing tracks**: ~10-30% of reconstructed tracks
- **Successful flash matches**: ~50-80% of quality tracks
- **Final training entries**: ~500-2000 per run

## Visualization and Debugging Tools

### Available Visualization Scripts
```bash
# Plot tracks and associated flashes
python visualizations/plot_tracks_flashes.py \
    --input cosmic_tracks_run3.root \
    --event-id 42 \
    --output track_flash_display.png

# Display flash-track matching results
python visualizations/plot_flash_matches.py \
    --input matched_flashes_run3.root \
    --entry-range 0:100 \
    --output matching_summary.png

# Show CRT hit correlations
python visualizations/display_crt_matches.py \
    --input matched_flashes_run3.root \
    --crt-system top_bottom \
    --output crt_correlation.png
```

### Debug Mode
Enable detailed logging and intermediate file outputs:
```bash
export FLASHMATCH_DEBUG=1
export FLASHMATCH_LOG_LEVEL=DEBUG

# Run with debug output
./build/installed/bin/flashmatch_dataprep \
    --input cosmic_tracks_run3.root \
    --output matched_flashes_run3.root \
    --debug-output debug_info.root \
    --verbose 2
```

## Configuration Management

All algorithm parameters are externalized in YAML configuration files to enable:
- Easy parameter tuning without recompilation
- Systematic studies of selection criteria
- Reproducible analysis configurations
- Student-friendly parameter exploration

### Configuration Files
- `config/quality_cuts.yaml`: Track selection criteria
- `config/flash_matching.yaml`: Flash-track matching parameters
- `config/output_format.yaml`: HDF5 schema and compression settings
- `config/cosmic_reconstruction.yaml`: Step 1 reconstruction parameters

## Student Development Guide

### Getting Started for New Students
1. **Environment Setup**: Follow the environment setup instructions above
2. **Build the Code**: Complete the build instructions
3. **Run a Test**: Process a single small file through the complete pipeline
4. **Explore Configurations**: Modify YAML parameters and observe effects
5. **Visualization**: Use plotting scripts to understand the data

### Common Development Tasks
- **Modify Quality Cuts**: Edit `src/CosmicTrackSelector.cxx` and `config/quality_cuts.yaml`
- **Improve Matching**: Enhance algorithms in `src/FlashTrackMatcher.cxx`
- **Add Visualizations**: Create new plotting scripts in `visualizations/`
- **Optimize Performance**: Profile and optimize C++ code for large-scale processing

### Debugging Tips
- Use `FLASHMATCH_DEBUG=1` for verbose output
- Check intermediate ROOT files with ROOT browser: `root -l debug_info.root`
- Validate configurations with: `python python/config_parser.py --validate config/`
- Monitor memory usage with: `valgrind --tool=massif ./flashmatch_dataprep`

## Performance Considerations

### Memory Usage
- Typical memory usage: 2-8 GB per processing job
- Large events may require up to 16 GB
- Use batch processing for datasets >100 files

### CPU Requirements
- Single-threaded C++ processing: ~1-5 minutes per cosmic ray file
- Parallel processing recommended for large datasets
- HDF5 conversion: ~30 seconds per matched event file

### Storage Requirements
- Input cosmic ray files: ~1-5 GB per run
- Output HDF5 training files: ~100-500 MB per run
- Intermediate debug files: ~500 MB - 2 GB per run

## Troubleshooting Common Issues

### Build Problems
```bash
# Missing ubdl environment
source /path/to/ubdl/setenv_py3_container.sh
source /path/to/ubdl/configure_container.sh

# Missing ROOT
export ROOTSYS=/path/to/root
source $ROOTSYS/bin/thisroot.sh

# CMake cannot find packages
export CMAKE_PREFIX_PATH=$LARLITE_LIBDIR:$LARCV_LIBDIR:$CMAKE_PREFIX_PATH
```

### Runtime Problems
```bash
# Segmentation faults
export FLASHMATCH_DEBUG=1
gdb ./build/installed/bin/flashmatch_dataprep

# Memory issues
ulimit -v 16777216  # Limit virtual memory to 16GB
export MALLOC_CHECK_=2

# File I/O errors
ls -la input_file.root  # Check file permissions
root -l -q input_file.root  # Verify ROOT file integrity
```

### Data Quality Issues
- **Low matching efficiency**: Adjust timing tolerances in `config/flash_matching.yaml`
- **Poor track quality**: Relax cuts in `config/quality_cuts.yaml`
- **CRT correlation problems**: Check CRT calibration and timing offsets

## Future Development Areas

### Algorithmic Improvements
- Machine learning-based flash-track matching
- Advanced cosmic ray track reconstruction
- Multi-track flash decomposition
- Real-time data quality monitoring

### Performance Optimizations
- Multi-threading for parallel processing
- GPU acceleration for geometric computations
- Streaming HDF5 I/O for large datasets
- Memory-mapped file access for huge files

### Analysis Extensions
- Systematic uncertainty quantification
- Cross-validation with simulation
- Integration with beam neutrino data
- Extension to multiple LAr detector technologies