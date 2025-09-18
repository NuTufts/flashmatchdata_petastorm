#!/bin/bash

# Build script for test programs
# This script helps build and test the ModelInputInterface

set -e  # Exit on error

echo "============================================="
echo "Building flashmatch_dataprep test programs"
echo "============================================="

# Check environment
if [ -z "$LIBTORCH_DIR" ]; then
    echo "Error: LIBTORCH_DIR not set"
    echo "Please source the environment setup:"
    echo "  source ../../setenv_flashmatchdata.sh"
    exit 1
fi

echo "Environment check:"
echo "  LIBTORCH_DIR: $LIBTORCH_DIR"
[ -n "$LARLITE_BASEDIR" ] && echo "  LARLITE: $LARLITE_BASEDIR"
[ -n "$LARCV_BASEDIR" ] && echo "  LARCV: $LARCV_BASEDIR"
[ -n "$LARFLOW_BASEDIR" ] && echo "  LARFLOW: $LARFLOW_BASEDIR"
echo ""

# Parse arguments
BUILD_MODE=${1:-debug}
TARGET=${2:-all}

echo "Build configuration:"
echo "  Mode: $BUILD_MODE"
echo "  Target: $TARGET"
echo ""

# Clean if requested
if [ "$TARGET" = "clean" ]; then
    echo "Cleaning..."
    make clean
    exit 0
fi

# Build
echo "Building..."
make BUILD_MODE=$BUILD_MODE $TARGET

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo ""

    # List built executables
    if [ -d "bin" ]; then
        echo "Built executables:"
        ls -la bin/
        echo ""
    fi

    # Provide usage instructions
    echo "To run tests:"
    echo "  ./bin/test_model_input_interface [siren_model.pt]"
    echo ""
    echo "To run example inference:"
    echo "  ./bin/example_siren_inference <siren_model.pt>"
    echo ""

    # Check if model exists
    if [ -f "../siren_model.pt" ]; then
        echo "Found SIREN model at ../siren_model.pt"
        echo ""
        echo "Quick test command:"
        echo "  ./bin/test_model_input_interface ../siren_model.pt"
    else
        echo "Note: No siren_model.pt found in parent directory."
        echo "Generate one with:"
        echo "  cd .."
        echo "  python make_siren_trace.py config_trace_siren_model.yaml"
    fi
else
    echo "Build failed!"
    exit 1
fi
