#!/bin/bash

# run_cosmic_reconstruction.sh
# Script to run cosmic ray reconstruction using larflow::reco::CosmicParticleReconstruction
# This implements Step 1 of the data preparation pipeline

# Default values
INPUT_DLMERGED=""
INPUT_LARFLOW=""
OUTPUT=""
NUM_ENTRIES=""
START_ENTRY=0
TICK_BACKWARDS=false
IS_MC=false
VERSION=2
LOG_LEVEL=1
VERBOSE=false
RUN_LARMATCH=false

# Parameters in case we need to run larmatch first
export OMP_NUM_THREADS=16
LARMATCH_DIR=${UBDL_BASEDIR}/larflow/larmatchnet/larmatch/
WEIGHTS_DIR=${LARMATCH_DIR}/
WEIGHT_FILE=checkpoint.easy-wave-79.93000th.tar
CONFIG_FILE=${LARMATCH_DIR}/standard_deploy_larmatchme_cpu.yaml
LARMATCHME_SCRIPT=${LARMATCH_DIR}/deploy_larmatchme_v2.py
# larmatch v2 (shower keypoint version)
#LARMATCHME_CMD="python3 ${LARMATCHME_SCRIPT} --config-file ${CONFIG_FILE} --input-larcv ${baseinput} --input-larlite ${baseinput} --weights ${WEIGHTS_DIR}/${WEIGHT_FILE} --output ${baselm} --min-score 0.3 --adc-name wire --device-name cpu --use-skip-limit --allow-output-overwrite -n ${NEVENTS}"
#echo $CMD



# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run cosmic ray reconstruction using larflow::reco::CosmicParticleReconstruction

Required Arguments:
  -i, --input-dlmerged FILE    Input dlmerged file (ADC, ssnet, badch images/info)
  -l, --input-larflow FILE     Input larflow file (larlite::larflow3dhit objects)
  -o, --output FILE            Output file name

Optional Arguments:
  -n, --num-entries N          Number of entries to process (default: all)
  -s, --start-entry N          Starting entry number (default: 0)
  -tb, --tick-backwards        Input larcv images are tick-backward (default: false)
  -mc, --is-mc                 Store MC information (default: false)
  -v, --version N              Reconstruction version (default: 2)
  -ll, --log-level N           Log verbosity 0=debug, 1=info, 2=normal, 3=warning, 4=error (default: 1)
  --verbose                    Enable verbose output from this script
  --run-larmatch               Run larmatch to generate larflow file before cosmic reconstruction
  -h, --help                   Display this help message

Examples:
  # Basic usage
  $0 -i dlmerged_cosmic.root -l larflow_cosmic.root -o cosmic_reco_output.root

  # Process only first 100 events with MC info
  $0 -i dlmerged_cosmic.root -l larflow_cosmic.root -o cosmic_reco_output.root -n 100 -mc

  # Process with tick-backwards and debug logging
  $0 -i dlmerged_cosmic.root -l larflow_cosmic.root -o cosmic_reco_output.root -tb -ll 0

  # Run larmatch first to generate larflow file
  $0 -i dlmerged_cosmic.root -l larflow_cosmic.root -o cosmic_reco_output.root --run-larmatch

Notes:
  - Requires ubdl environment to be set up
  - Input files must contain proper data products
  - Output file will contain cosmic ray tracks with flash and CRT information
  - For Runs 3+, CRT information will be included automatically

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input-dlmerged)
            INPUT_DLMERGED="$2"
            shift 2
            ;;
        -l|--input-larflow)
            INPUT_LARFLOW="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -n|--num-entries)
            NUM_ENTRIES="$2"
            shift 2
            ;;
        -s|--start-entry)
            START_ENTRY="$2"
            shift 2
            ;;
        -tb|--tick-backwards)
            TICK_BACKWARDS=true
            shift
            ;;
        -mc|--is-mc)
            IS_MC=true
            shift
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -ll|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --run-larmatch)
            RUN_LARMATCH=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_DLMERGED" ]]; then
    echo -e "${RED}Error: Input dlmerged file is required${NC}"
    usage
    exit 1
fi

if [[ -z "$INPUT_LARFLOW" ]]; then
    if [[ "$RUN_LARMATCH" == false ]]; then
        echo -e "${RED}Error: Input larflow file is required (or use --run-larmatch to generate it)${NC}"
        usage
        exit 1
    fi
fi

if [[ -z "$OUTPUT" ]]; then
    echo -e "${RED}Error: Output file is required${NC}"
    usage
    exit 1
fi

# Check if input files exist
if [[ ! -f "$INPUT_DLMERGED" ]]; then
    echo -e "${RED}Error: Input dlmerged file does not exist: $INPUT_DLMERGED${NC}"
    exit 1
fi

if [[ "$RUN_LARMATCH" == false ]] && [[ ! -f "$INPUT_LARFLOW" ]]; then
    echo -e "${RED}Error: Input larflow file does not exist: $INPUT_LARFLOW${NC}"
    exit 1
fi

# Check if output file already exists
if [[ -f "$OUTPUT" ]]; then
    echo -e "${YELLOW}Warning: Output file already exists: $OUTPUT${NC}"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Aborted.${NC}"
        exit 1
    fi
    rm -f "$OUTPUT"
fi

# Generate larflow filename if needed
if [[ "$RUN_LARMATCH" == true ]] && [[ -z "$INPUT_LARFLOW" ]]; then
    # Generate larflow filename from dlmerged filename
    DLMERGED_BASENAME=$(basename "$INPUT_DLMERGED" .root)
    DLMERGED_DIR=$(dirname "$INPUT_DLMERGED")
    INPUT_LARFLOW="${DLMERGED_DIR}/${DLMERGED_BASENAME}_larmatch.root"
    echo -e "${YELLOW}Generated larflow filename: $INPUT_LARFLOW${NC}"
fi

# Create log directory if it doesn't exist
LOG_DIR="$(dirname "$OUTPUT")/logs"
mkdir -p "$LOG_DIR"

# Generate output basename for log files
OUTPUT_BASENAME=$(basename "$OUTPUT" .root)

# Verbose output
if [[ "$VERBOSE" == true ]]; then
    echo -e "${BLUE}Configuration:${NC}"
    echo "  Input dlmerged: $INPUT_DLMERGED"
    echo "  Input larflow:  $INPUT_LARFLOW"
    echo "  Output:         $OUTPUT"
    echo "  Num entries:    ${NUM_ENTRIES:-all}"
    echo "  Start entry:    $START_ENTRY"
    echo "  Tick backwards: $TICK_BACKWARDS"
    echo "  Is MC:          $IS_MC"
    echo "  Version:        $VERSION"
    echo "  Log level:      $LOG_LEVEL"
    echo "  Run larmatch:   $RUN_LARMATCH"
    echo ""
fi

# Check ubdl environment
echo -e "${YELLOW}Checking ubdl environment...${NC}"
if [[ -z "$LARFLOW_BASEDIR" ]]; then
    echo -e "${RED}Error: LARFLOW_BASEDIR not set. Please set up ubdl environment first:${NC}"
    echo "  source setenv_py3_container.sh"
    echo "  source configure_container.sh"
    exit 1
fi

# Check for required spline files
SPLINE_FILE="$LARFLOW_BASEDIR/larflow/Reco/data/Proton_Muon_Range_dEdx_LAr_TSplines.root"
if [[ ! -f "$SPLINE_FILE" ]]; then
    echo -e "${RED}Error: Required spline file not found: $SPLINE_FILE${NC}"
    echo "Please ensure larflow is properly built and data files are present"
    exit 1
fi

# Run larmatch if requested
if [[ "$RUN_LARMATCH" == true ]]; then
    echo -e "${GREEN}Running larmatch to generate larflow file...${NC}"
    
    # Check if larmatch files exist
    if [[ ! -f "$WEIGHTS_DIR/$WEIGHT_FILE" ]]; then
        echo -e "${RED}Error: Larmatch weights file not found: $WEIGHTS_DIR/$WEIGHT_FILE${NC}"
        exit 1
    fi
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}Error: Larmatch config file not found: $CONFIG_FILE${NC}"
        exit 1
    fi
    
    if [[ ! -f "$LARMATCHME_SCRIPT" ]]; then
        echo -e "${RED}Error: Larmatch script not found: $LARMATCHME_SCRIPT${NC}"
        exit 1
    fi
    
    # Build larmatch command
    LARMATCHME_CMD="python3 ${LARMATCHME_SCRIPT}"
    LARMATCHME_CMD="$LARMATCHME_CMD --config-file ${CONFIG_FILE}"
    LARMATCHME_CMD="$LARMATCHME_CMD --input-larcv ${INPUT_DLMERGED}"
    LARMATCHME_CMD="$LARMATCHME_CMD --input-larlite ${INPUT_DLMERGED}"
    LARMATCHME_CMD="$LARMATCHME_CMD --weights ${WEIGHTS_DIR}/${WEIGHT_FILE}"
    LARMATCHME_CMD="$LARMATCHME_CMD --output ${INPUT_LARFLOW}"
    LARMATCHME_CMD="$LARMATCHME_CMD --min-score 0.3"
    LARMATCHME_CMD="$LARMATCHME_CMD --adc-name wire"
    LARMATCHME_CMD="$LARMATCHME_CMD --device-name cpu"
    LARMATCHME_CMD="$LARMATCHME_CMD --use-skip-limit"
    LARMATCHME_CMD="$LARMATCHME_CMD --allow-output-overwrite"
    
    if [[ -n "$NUM_ENTRIES" ]]; then
        LARMATCHME_CMD="$LARMATCHME_CMD -n ${NUM_ENTRIES}"
    fi
    
    # Create log for larmatch
    LARMATCH_LOG_FILE="$LOG_DIR/larmatch_${OUTPUT_BASENAME}_$(date +%Y%m%d_%H%M%S).log"
    
    echo -e "${BLUE}Larmatch command: $LARMATCHME_CMD${NC}"
    echo -e "${BLUE}Larmatch log file: $LARMATCH_LOG_FILE${NC}"
    
    # Save original LD_LIBRARY_PATH
    ORIG_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    
    # Remove libtorch from LD_LIBRARY_PATH for larmatch to avoid conflict with pip-installed pytorch
    LARMATCH_LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "libtorch" | tr '\n' ':' | sed 's/:$//')
    
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${YELLOW}Temporarily removing libtorch from LD_LIBRARY_PATH for larmatch${NC}"
    fi
    
    # Run larmatch with modified environment
    if [[ "$VERBOSE" == true ]]; then
        LD_LIBRARY_PATH="$LARMATCH_LD_LIBRARY_PATH" eval "$LARMATCHME_CMD" 2>&1 | tee "$LARMATCH_LOG_FILE"
        LARMATCH_RESULT=${PIPESTATUS[0]}
    else
        LD_LIBRARY_PATH="$LARMATCH_LD_LIBRARY_PATH" eval "$LARMATCHME_CMD" > "$LARMATCH_LOG_FILE" 2>&1
        LARMATCH_RESULT=$?
    fi
    
    # Restore original LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$ORIG_LD_LIBRARY_PATH"
    
    # Check larmatch result
    if [[ $LARMATCH_RESULT -ne 0 ]]; then
        echo -e "${RED}✗ Larmatch failed with exit code $LARMATCH_RESULT${NC}"
        echo -e "${YELLOW}Check log file for details: $LARMATCH_LOG_FILE${NC}"
        
        if [[ -f "$LARMATCH_LOG_FILE" ]]; then
            echo ""
            echo -e "${YELLOW}Last 10 lines of larmatch log:${NC}"
            tail -n 10 "$LARMATCH_LOG_FILE"
        fi
        
        exit $LARMATCH_RESULT
    fi
    
    echo -e "${GREEN}✓ Larmatch completed successfully!${NC}"
    echo -e "${BLUE}Generated larflow file: $INPUT_LARFLOW${NC}"
    echo ""
fi

# Build the python command
PYTHON_CMD="python3 $LARFLOW_BASEDIR/larflow/Reco/test/run_cosmicreco.py"
PYTHON_CMD="$PYTHON_CMD -i \"$INPUT_DLMERGED\""
PYTHON_CMD="$PYTHON_CMD -l \"$INPUT_LARFLOW\""
PYTHON_CMD="$PYTHON_CMD -o \"$OUTPUT\""
PYTHON_CMD="$PYTHON_CMD -e $START_ENTRY"
PYTHON_CMD="$PYTHON_CMD -v $VERSION"
PYTHON_CMD="$PYTHON_CMD -ll $LOG_LEVEL"

if [[ -n "$NUM_ENTRIES" ]]; then
    PYTHON_CMD="$PYTHON_CMD -n $NUM_ENTRIES"
fi

if [[ "$TICK_BACKWARDS" == true ]]; then
    PYTHON_CMD="$PYTHON_CMD -tb"
fi

if [[ "$IS_MC" == true ]]; then
    PYTHON_CMD="$PYTHON_CMD -mc"
fi

# Generate log file name
LOG_FILE="$LOG_DIR/cosmic_reco_${OUTPUT_BASENAME}_$(date +%Y%m%d_%H%M%S).log"

echo -e "${GREEN}Starting cosmic ray reconstruction...${NC}"
echo -e "${BLUE}Command: $PYTHON_CMD${NC}"
echo -e "${BLUE}Log file: $LOG_FILE${NC}"
echo ""

# Run the reconstruction
if [[ "$VERBOSE" == true ]]; then
    eval "$PYTHON_CMD" 2>&1 | tee "$LOG_FILE"
    RESULT=${PIPESTATUS[0]}
else
    eval "$PYTHON_CMD" > "$LOG_FILE" 2>&1
    RESULT=$?
fi

# Check result
if [[ $RESULT -eq 0 ]]; then
    echo -e "${GREEN}✓ Cosmic ray reconstruction completed successfully!${NC}"
    echo -e "${BLUE}Output file: $OUTPUT${NC}"
    echo -e "${BLUE}Log file: $LOG_FILE${NC}"
    
    # Show basic file info
    if command -v root &> /dev/null; then
        echo ""
        echo -e "${YELLOW}Output file information:${NC}"
        root -l -q -b "$OUTPUT" -e "gFile->ls(); gFile->Close();" 2>/dev/null | grep -E "(TTree|entries)"
    fi
else
    echo -e "${RED}✗ Cosmic ray reconstruction failed with exit code $RESULT${NC}"
    echo -e "${YELLOW}Check log file for details: $LOG_FILE${NC}"
    
    # Show last few lines of log for quick debugging
    if [[ -f "$LOG_FILE" ]]; then
        echo ""
        echo -e "${YELLOW}Last 10 lines of log:${NC}"
        tail -n 10 "$LOG_FILE"
    fi
    
    exit $RESULT
fi

echo ""
echo -e "${GREEN}Ready for next step: quality cuts and flash-track matching${NC}"
echo -e "${BLUE}Next command:${NC}"
echo "  ./build/installed/bin/flashmatch_dataprep --input \"$OUTPUT\" --output matched_output.root"
