# Test Programs for ModelInputInterface

This directory contains test programs for the C++ ModelInputInterface that prepares input tensors for the SIREN flash matching model.

## Building with GNUmakefile

### Prerequisites

1. Source the environment:
```bash
cd ../..
source setenv_flashmatchdata.sh
```

2. Ensure you have the SIREN model traced to TorchScript:
```bash
cd ..
python make_siren_trace.py config_trace_siren_model.yaml
```

### Build Commands

Build all test programs:
```bash
make all
# or
make tests
```

Build specific test:
```bash
make test_model_input_interface
```

Build with debug mode (default):
```bash
make BUILD_MODE=debug all
```

Build with release mode (optimized):
```bash
make BUILD_MODE=release all
```

Clean and rebuild:
```bash
make rebuild
```

Show build configuration:
```bash
make info
```

### Using the Build Script

For convenience, use the build script:
```bash
./build_test.sh           # Build all in debug mode
./build_test.sh release   # Build all in release mode
./build_test.sh debug test_model_input_interface  # Build specific test
./build_test.sh clean     # Clean build artifacts
```

## Running Tests

### test_model_input_interface

Tests the input tensor preparation:
```bash
# Without model (just tests tensor preparation)
./bin/test_model_input_interface

# With SIREN model (tests full pipeline)
./bin/test_model_input_interface ../siren_model.pt
```

This test:
- Creates dummy hit data
- Prepares input tensors using ModelInputInterface
- Validates tensor shapes and dimensions
- Optionally runs inference with a TorchScript model

### example_siren_inference

Example of complete inference pipeline:
```bash
./bin/example_siren_inference ../siren_model.pt
```

This example:
- Shows how to set up the interface
- Prepares real-shaped input tensors
- Loads and runs the SIREN model
- Processes output to get PMT predictions

## Expected Output

When running with a model, you should see:
- Input tensor shapes: `voxel_features` (N*32, 7) and `voxel_charge` (N*32, 1)
- Model output shape matching input
- Predicted PE values for each of the 32 PMTs
- Total predicted PE sum

## Troubleshooting

### Linking Errors
If you get ABI-related linking errors:
1. Check that LibTorch matches your compiler's ABI setting
2. The makefile sets `-D_GLIBCXX_USE_CXX11_ABI=0` by default
3. Adjust if your LibTorch uses CXX11 ABI

### Missing Dependencies
- Ensure all environment variables are set (LIBTORCH_DIR, etc.)
- Run `make info` to see detected paths
- The build will warn about missing optional dependencies

### Runtime Errors
- Check that the model file path is correct
- Ensure the model was traced with compatible PyTorch version
- Verify tensor dimensions match model expectations

## Files

### Source Files
- `src/ModelInputInterface.cxx` - Main implementation
- `src/PrepareVoxelOutput.cxx` - Voxel preparation utilities
- `src/PMTPositions.cxx` - PMT geometry definitions

### Header Files
- `include/ModelInputInterface.h` - Main interface header
- `include/PMTPositions.h` - PMT position utilities
- `include/PrepareVoxelOutput.h` - Voxel output preparation

### Test Files
- `test_model_input_interface.cxx` - Unit test for interface
- `example_siren_inference.cxx` - Example usage

### Build Files
- `GNUmakefile` - Standalone makefile for tests
- `CMakeLists.txt` - Main CMake configuration
- `build_test.sh` - Convenience build script