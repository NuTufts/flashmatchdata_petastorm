#!/bin/bash

# setup_environment.sh
# Environment setup script for flashmatch data preparation pipeline
# This script should be sourced after setting up the ubdl environment

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up flashmatch data preparation environment...${NC}"

# Check if we're in the correct directory
if [[ ! -f "CMakeLists.txt" ]]; then
    echo -e "${RED}Error: Please run this script from the data_prep directory${NC}"
    return 1
fi

# Set the data_prep base directory
export FLASHMATCH_DATAPREP_DIR="$(pwd)"
export FLASHMATCH_DATAPREP_BUILD_DIR="${FLASHMATCH_DATAPREP_DIR}/build"
export FLASHMATCH_DATAPREP_INSTALL_DIR="${FLASHMATCH_DATAPREP_BUILD_DIR}/installed"

# Add to PATH
export PATH="${FLASHMATCH_DATAPREP_INSTALL_DIR}/bin:${PATH}"

# Add to library path
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LD_LIBRARY_PATH="${FLASHMATCH_DATAPREP_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${FLASHMATCH_DATAPREP_INSTALL_DIR}/lib:${DYLD_LIBRARY_PATH}"
fi

# Python path for utilities
export PYTHONPATH="${FLASHMATCH_DATAPREP_DIR}/python:${PYTHONPATH}"

# Configuration directory
export FLASHMATCH_CONFIG_DIR="${FLASHMATCH_DATAPREP_DIR}/config"

# Create necessary directories if they don't exist
mkdir -p "${FLASHMATCH_DATAPREP_BUILD_DIR}"
mkdir -p "${FLASHMATCH_DATAPREP_DIR}/config"
mkdir -p "${FLASHMATCH_DATAPREP_DIR}/logs"
mkdir -p "${FLASHMATCH_DATAPREP_DIR}/output"

# Verify ubdl environment is set up
echo -e "${YELLOW}Checking ubdl environment...${NC}"

required_vars=("LARLITE_LIBDIR" "LARCV_LIBDIR" "LARFLOW_LIBDIR" "UBLARCVAPP_LIBDIR")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo -e "${RED}Error: Missing required ubdl environment variables:${NC}"
    for var in "${missing_vars[@]}"; do
        echo -e "${RED}  - $var${NC}"
    done
    echo -e "${YELLOW}Please run the following first:${NC}"
    echo "  source setenv_py3_container.sh"
    echo "  source configure_container.sh"
    return 1
fi

# Check for ROOT
if ! command -v root &> /dev/null; then
    echo -e "${RED}Error: ROOT not found in PATH${NC}"
    echo -e "${YELLOW}Please ensure ROOT is properly installed and configured${NC}"
    return 1
fi

# Check for Python packages
echo -e "${YELLOW}Checking Python packages...${NC}"
python3 -c "import h5py, numpy, yaml" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo -e "${RED}Error: Missing required Python packages (h5py, numpy, yaml)${NC}"
    echo -e "${YELLOW}Please install with: pip install h5py numpy pyyaml${NC}"
    return 1
fi

# Set debug options (can be overridden by user)
export FLASHMATCH_DEBUG=${FLASHMATCH_DEBUG:-0}
export FLASHMATCH_LOG_LEVEL=${FLASHMATCH_LOG_LEVEL:-INFO}

# Create a build status function
function flashmatch_build_status() {
    if [[ -f "${FLASHMATCH_DATAPREP_INSTALL_DIR}/bin/flashmatch_dataprep" ]]; then
        echo -e "${GREEN}✓ C++ program built and installed${NC}"
    else
        echo -e "${YELLOW}⚠ C++ program not built. Run: cd build && cmake .. && make -j4 && make install${NC}"
    fi
    
    if [[ -f "${FLASHMATCH_DATAPREP_DIR}/config/quality_cuts.yaml" ]]; then
        echo -e "${GREEN}✓ Configuration files present${NC}"
    else
        echo -e "${YELLOW}⚠ Configuration files missing. Create config/*.yaml files${NC}"
    fi
}

# Create a quick test function
function flashmatch_test_environment() {
    echo -e "${YELLOW}Testing environment setup...${NC}"
    
    # Test ROOT
    root -l -q -b &>/dev/null
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ ROOT is working${NC}"
    else
        echo -e "${RED}✗ ROOT test failed${NC}"
    fi
    
    # Test Python imports
    python3 -c "import ROOT, h5py, numpy, yaml" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Python packages are working${NC}"
    else
        echo -e "${RED}✗ Python package test failed${NC}"
    fi
    
    # Test ubdl libraries
    python3 -c "import larlite, larcv" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ ubdl Python bindings are working${NC}"
    else
        echo -e "${RED}✗ ubdl Python bindings test failed${NC}"
    fi
}

# Show status
echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${YELLOW}Environment variables set:${NC}"
echo "  FLASHMATCH_DATAPREP_DIR = ${FLASHMATCH_DATAPREP_DIR}"
echo "  FLASHMATCH_DATAPREP_BUILD_DIR = ${FLASHMATCH_DATAPREP_BUILD_DIR}"
echo "  FLASHMATCH_DATAPREP_INSTALL_DIR = ${FLASHMATCH_DATAPREP_INSTALL_DIR}"
echo "  FLASHMATCH_CONFIG_DIR = ${FLASHMATCH_CONFIG_DIR}"
echo ""
echo -e "${YELLOW}Available commands:${NC}"
echo "  flashmatch_build_status    - Check build status"
echo "  flashmatch_test_environment - Test environment"
echo ""

# Run initial status check
flashmatch_build_status