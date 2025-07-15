# Cosmic Ray Track-Flash Timing Dashboard

Interactive Plotly Dash visualization for analyzing timing correlations between cosmic ray tracks and optical flashes in MicroBooNE LArTPC data.

## Purpose

This educational tool helps students understand the temporal relationship between cosmic ray particle tracks and optical flashes for flash-track matching studies. The visualization shows how particle tracks (derived from TPC wire signals) correlate in time with optical flashes (from PMT signals).

## Features

- **Timing Correlation Plot**: Single 2D plot showing tracks and flashes vs. Z-position and time
- **Track Visualization**: Cosmic ray tracks plotted as Z-coordinate vs. drift time
- **Flash Visualization**: Optical flashes shown as horizontal lines at flash times
- **TPC Readout Window**: Bounds showing the time range of available TPC data
- **Cathode Crossing**: Additional flash timing for particles crossing at the cathode
- **Educational Focus**: Clear visual correlation between particle timing and light production

## Physics Concepts

### Drift Time Calculation
- **X-coordinate to time conversion**: `time = x_position / drift_velocity`
- **Drift velocity**: 0.109 cm/μs (argon at MicroBooNE conditions)
- **Physics**: Ionization electrons drift toward wire planes, X-position indicates drift time

### TPC Readout Window
- **Start time**: -400 μs (relative to trigger)
- **End time**: +2635 μs (relative to trigger)
- **Significance**: Only tracks within this window can be reconstructed from TPC data

### Flash-Track Matching
- **Spatial correlation**: Flash Z-position should align with track Z-trajectory
- **Temporal correlation**: Flash time should match track drift time
- **Cathode crossing**: Particles may produce light when crossing the cathode plane

## Requirements

- Python 3.7+
- ROOT/PyROOT (available in the container environment)
- Required Python packages (see `requirements.txt`)

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure ROOT is available:**
   ```bash
   # In the container environment
   source setenv_py3_container.sh
   source configure_container.sh
   ```

## Usage

### Basic Usage

```bash
# Run with cosmic reconstruction data
python cosmic_dashboard.py --input cosmic_reco_output.root --port 8050

# Run with dummy data for testing
python cosmic_dashboard.py --port 8050
```

### Command Line Options

- `--input, -i`: Path to ROOT file with FlashMatchData tree
- `--host`: Host to run dashboard on (default: 127.0.0.1)  
- `--port, -p`: Port to run dashboard on (default: 8050)
- `--debug`: Run in debug mode with auto-reload

### Examples

```bash
# Run with specific cosmic reconstruction output
python cosmic_dashboard.py -i test_cosmicreco.root

# Run on different port
python cosmic_dashboard.py -i cosmic_data.root --port 8051

# Run in debug mode for development
python cosmic_dashboard.py --debug
```

## Data Format

The dashboard expects ROOT files containing a `FlashMatchData` TTree with the following branches:

### FlashMatchData Tree Structure
- **Event info**: `run`, `subrun`, `event`
- **track_v**: Vector of `larlite::track` objects (cosmic ray tracks)
- **opflash_v**: Vector of `larlite::opflash` objects (optical flashes)
- **crttrack_v**: Vector of `larlite::crttrack` objects (CRT tracks)
- **crthit_v**: Vector of `larlite::crthit` objects (CRT hits)

### Track Data
- **Trajectory points**: 3D positions along particle path
- **Start/End positions**: Track endpoints (using `Vertex()` and `End()` methods)
- **Track length**: Total path length through detector

### Flash Data
- **Flash timing**: Time relative to trigger (`Time()` method)
- **PMT signals**: Photoelectron counts per PMT (`PE(pmt_id)` method)
- **Flash position**: Y and Z center coordinates (`YCenter()`, `ZCenter()` methods)
- **Total PE**: Sum of all PMT signals (`TotalPE()` method)

## Visualization Components

### Main Timing Plot

**X-axis**: Z Position [cm] (0-1037 cm, MicroBooNE detector range)
**Y-axis**: Time from Trigger [μs]

### Track Representation (Blue)
- **Points and lines**: Connected trajectory showing particle path through time-space
- **Time calculation**: X-coordinate converted using drift velocity
- **Hover info**: Track number, Z-position, calculated drift time

### Flash Representation (Red)
- **Horizontal lines**: Solid lines at flash time with width showing Z-extent
- **Flash centers**: Diamond markers at flash center positions
- **Cathode crossing**: Dashed lines showing expected time if particle crossed cathode
- **Hover info**: Flash number, time, total PE, Z-position

### TPC Readout Window (Black Dashed)
- **Lower bound**: -400 μs line showing start of TPC readout
- **Upper bound**: +2635 μs line showing end of TPC readout
- **Educational value**: Shows which tracks/flashes can be correlated

## Interactive Features

### Event Navigation
- **Dropdown menu**: Select event number from available entries
- **Load button**: Trigger data loading (prevents accidental real-time updates)
- **Event counter**: Shows total number of events in file

### Hover Information
- **Track details**: Track ID, Z-position, drift time
- **Flash details**: Flash ID, time, PE count, Z-position
- **Window bounds**: TPC readout start/end times

## Educational Use Cases

### Understanding Flash-Track Matching
1. **Good matches**: Tracks and flashes at similar times and Z-positions
2. **Poor matches**: Temporal or spatial misalignment
3. **Out-of-time flashes**: Flashes outside TPC readout window
4. **Multiple tracks**: Events with several cosmic rays

### Physics Learning Objectives
- **Drift physics**: How X-position relates to time via electron drift
- **Detector response**: TPC vs. PMT timing and spatial information
- **Coincidence analysis**: Correlating different detector subsystems
- **Background rejection**: Identifying cosmic rays vs. beam events

## Troubleshooting

### Common Issues

1. **ROOT not found**: Ensure container environment is properly sourced
2. **FlashMatchData tree missing**: Verify cosmic reconstruction completed successfully
3. **Empty plots**: Check that selected event contains track/flash data
4. **Connection refused**: Verify port is not already in use

### Debug Information

The dashboard prints helpful information on startup:
- Available ROOT trees in file
- Number of entries in FlashMatchData tree
- Track and flash object methods (for debugging data access)

### Fallback Mode

If ROOT files cannot be loaded, the dashboard uses dummy data for interface testing.

## File Structure

```
visualizations/
├── cosmic_dashboard.py     # Main timing correlation dashboard
├── plot_flash_matches.py   # Static matplotlib visualizations
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Integration with Analysis Pipeline

The dashboard works with output from:
1. **Cosmic reconstruction**: `run_cosmic_reconstruction.sh` script
2. **FlashMatchData tree**: Created by `CosmicParticleReconstruction` class
3. **LArFlow pipeline**: 3D reconstruction from 2D wire signals

## Performance Notes

- **Button-triggered updates**: Prevents slow real-time data loading
- **Single event loading**: Efficient memory usage for large files
- **Browser compatibility**: Optimized for Chrome/Firefox

## Educational Context

This tool is designed for students learning:
- **LArTPC physics**: Liquid argon time projection chamber operation
- **Multi-detector systems**: Correlating TPC and PMT information
- **Particle reconstruction**: From raw signals to physics objects
- **Data analysis**: Interactive exploration of detector data

## Support

For issues related to:
- **Dashboard functionality**: Check this README and try debug mode
- **ROOT file format**: Verify FlashMatchData tree structure
- **Container environment**: Ensure ubdl environment is properly configured
- **Physics questions**: Consult MicroBooNE reconstruction documentation

Generated with Claude Code for MicroBooNE cosmic ray timing analysis.