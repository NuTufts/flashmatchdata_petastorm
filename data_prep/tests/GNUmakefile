# GNUmakefile for flashmatch_dataprep test programs
# Standalone makefile for building test executables without CMake

# ===================================================================
# CONFIGURATION SECTION
# ===================================================================

# Compiler
CXX := g++

# C++ Standard (match ROOT's standard)
CXX_STD := -std=c++14

# Build mode (debug or release)
BUILD_MODE ?= debug

# ===================================================================
# ENVIRONMENT VARIABLES (from ubdl setup)
# ===================================================================

# Check for required environment variables
ifndef LIBTORCH_DIR
$(error LIBTORCH_DIR is not set. Please source setenv_flashmatchdata.sh)
endif

ifndef LARLITE_BASEDIR
$(warning LARLITE_BASEDIR not set, some features may be disabled)
endif

ifndef LARCV_BASEDIR
$(warning LARCV_BASEDIR not set, some features may be disabled)
endif

ifndef LARFLOW_BASEDIR
$(warning LARFLOW_BASEDIR not set, some features may be disabled)
endif

# ===================================================================
# DIRECTORIES
# ===================================================================

# Source directories
SRC_DIR := ../src
INC_DIR := ../include
OBJ_DIR := obj
BIN_DIR := bin
TEST_DIR := .

# Create directories if they don't exist
$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(BIN_DIR))

# ===================================================================
# INCLUDES
# ===================================================================

# Project includes
INCLUDES := -I$(INC_DIR)

# ROOT includes
ROOT_CONFIG := root-config
ROOT_CFLAGS := $(shell $(ROOT_CONFIG) --cflags)
ROOT_INCLUDES := $(ROOT_CFLAGS)

# LibTorch includes
TORCH_INCLUDES := -I$(LIBTORCH_DIR)/include \
                  -I$(LIBTORCH_DIR)/include/torch/csrc/api/include

# UBDL includes (if available)
ifdef LARLITE_BASEDIR
INCLUDES += -I$(LARLITE_BASEDIR) \
            -I$(LARLITE_COREDIR)/Base \
            -I$(LARLITE_COREDIR)/DataFormat
endif

ifdef LARCV_BASEDIR
INCLUDES += -I$(LARCV_BASEDIR) \
            -I$(LARCV_COREDIR) \
            -I$(LARCV_APPDIR)
endif

ifdef LARFLOW_INCDIR
INCLUDES += -I$(LARFLOW_INCDIR)
endif

ifdef UBLARCVAPP_INCDIR
INCLUDES += -I$(UBLARCVAPP_INCDIR)
endif

# Python/NumPy includes and libraries (for any Python bindings)
PYTHON_CONFIG := python3-config
PYTHON_INCLUDES := $(shell $(PYTHON_CONFIG) --includes 2>/dev/null || echo "")
NUMPY_INCLUDES := $(shell python3 -c "import numpy; print('-I' + numpy.get_include())" 2>/dev/null || echo "")
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags 2>/dev/null || echo "")
PYTHON_LIBS := $(shell $(PYTHON_CONFIG) --libs 2>/dev/null || echo "-lpython3.8")

# ===================================================================
# LIBRARIES
# ===================================================================

# Library paths
LDFLAGS := -L$(LIBTORCH_DIR)/lib $(PYTHON_LDFLAGS)

ifdef LARLITE_LIBDIR
LDFLAGS += -L$(LARLITE_LIBDIR)
endif

ifdef LARCV_LIBDIR
LDFLAGS += -L$(LARCV_LIBDIR)
endif

ifdef LARFLOW_LIBDIR
LDFLAGS += -L$(LARFLOW_LIBDIR)
endif

ifdef UBLARCVAPP_LIBDIR
LDFLAGS += -L$(UBLARCVAPP_LIBDIR)
endif

# ROOT libraries
ROOT_LIBS := $(shell $(ROOT_CONFIG) --libs)

# LibTorch libraries
TORCH_LIBS := -ltorch -ltorch_cpu -lc10

# UBDL libraries (if available)
UBDL_LIBS :=

ifdef LARFLOW_LIBDIR
UBDL_LIBS += -lLArFlow_Voxelizer
endif

ifdef LARCV_LIBDIR
UBDL_LIBS += -lLArCVCoreBase -lLArCVCoreDataFormat
endif

ifdef LARLITE_LIBDIR
UBDL_LIBS += -llarlite_DataFormat -llarlite_LArUtil -llarlite_Base 
endif

# System libraries
SYS_LIBS := -lpthread -ldl

# Python libraries (needed by UBDL libraries with Python bindings)
PYTHON_LINK_LIBS := $(PYTHON_LIBS)

# All libraries
LIBS := $(TORCH_LIBS) $(ROOT_LIBS) $(UBDL_LIBS) $(PYTHON_LINK_LIBS) $(SYS_LIBS) -lpython3.8

# Runtime library paths
RPATH_FLAGS := -Wl,-rpath,$(LIBTORCH_DIR)/lib

ifdef LARLITE_LIBDIR
RPATH_FLAGS += -Wl,-rpath,$(LARLITE_LIBDIR)
endif

ifdef LARCV_LIBDIR
RPATH_FLAGS += -Wl,-rpath,$(LARCV_LIBDIR)
endif

ifdef LARFLOW_LIBDIR
RPATH_FLAGS += -Wl,-rpath,$(LARFLOW_LIBDIR)
endif

# ===================================================================
# COMPILER FLAGS
# ===================================================================

# Base flags
CXXFLAGS := $(CXX_STD) -Wall -Wextra -Wno-deprecated-copy -Wno-ignored-qualifiers -Wno-reorder -Wno-unused-parameter

# Build mode specific flags
ifeq ($(BUILD_MODE),debug)
    CXXFLAGS += -g -O0 -DDEBUG
else ifeq ($(BUILD_MODE),release)
    CXXFLAGS += -O3 -DNDEBUG
else
    $(error Invalid BUILD_MODE: $(BUILD_MODE). Use 'debug' or 'release')
endif

# LibTorch requires specific flags
CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1

# Add all includes
CXXFLAGS += $(INCLUDES) $(ROOT_INCLUDES) $(TORCH_INCLUDES) $(PYTHON_INCLUDES) $(NUMPY_INCLUDES)

# ===================================================================
# SOURCE FILES
# ===================================================================

# Main library sources
LIB_SOURCES := $(SRC_DIR)/ModelInputInterface.cxx \
               $(SRC_DIR)/PrepareVoxelOutput.cxx \
               $(SRC_DIR)/PMTPositions.cxx

# Test program sources
TEST_SOURCES := $(TEST_DIR)/test_model_input_interface.cxx

# Additional test programs (uncomment as needed)
# TEST_SOURCES += $(TEST_DIR)/example_siren_inference.cxx

# Object files
LIB_OBJECTS := $(patsubst $(SRC_DIR)/%.cxx,$(OBJ_DIR)/%.o,$(LIB_SOURCES))
TEST_OBJECTS := $(patsubst $(TEST_DIR)/%.cxx,$(OBJ_DIR)/%.o,$(TEST_SOURCES))

# Test executables
TEST_TARGETS := $(patsubst $(TEST_DIR)/%.cxx,$(BIN_DIR)/%,$(TEST_SOURCES))

# ===================================================================
# TARGETS
# ===================================================================

# Default target
.PHONY: all
all: tests

# Build all test programs
.PHONY: tests
tests: $(TEST_TARGETS)

# Build specific test program
.PHONY: test_model_input_interface
test_model_input_interface: $(BIN_DIR)/test_model_input_interface

# Build example inference program
.PHONY: example_siren_inference
example_siren_inference: $(BIN_DIR)/example_siren_inference

# ===================================================================
# BUILD RULES
# ===================================================================

# Compile library source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cxx
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test source files
$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cxx
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link test executables
$(BIN_DIR)/%: $(OBJ_DIR)/%.o $(LIB_OBJECTS)
	@echo "Linking $@..."
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(RPATH_FLAGS)

# Special target for test_model_input_interface with all dependencies
$(BIN_DIR)/test_model_input_interface: $(OBJ_DIR)/test_model_input_interface.o $(LIB_OBJECTS)
	@echo "Linking test_model_input_interface..."
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(RPATH_FLAGS)
	@echo "Built: $@"
	@echo "To run: $@ [path_to_siren_model.pt]"

# Special target for example_siren_inference
$(BIN_DIR)/example_siren_inference: $(OBJ_DIR)/example_siren_inference.o $(LIB_OBJECTS)
	@echo "Linking example_siren_inference..."
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS) $(RPATH_FLAGS)
	@echo "Built: $@"
	@echo "To run: $@ <path_to_siren_model.pt>"

# ===================================================================
# UTILITY TARGETS
# ===================================================================

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Clean and rebuild
.PHONY: rebuild
rebuild: clean all

# Show configuration
.PHONY: info
info:
	@echo "====================================="
	@echo "Build Configuration:"
	@echo "====================================="
	@echo "CXX:           $(CXX)"
	@echo "CXX_STD:       $(CXX_STD)"
	@echo "BUILD_MODE:    $(BUILD_MODE)"
	@echo ""
	@echo "Directories:"
	@echo "LIBTORCH_DIR:  $(LIBTORCH_DIR)"
ifdef LARLITE_BASEDIR
	@echo "LARLITE:       $(LARLITE_BASEDIR)"
endif
ifdef LARCV_BASEDIR
	@echo "LARCV:         $(LARCV_BASEDIR)"
endif
ifdef LARFLOW_DIR
	@echo "LARFLOW:       $(LARFLOW_DIR)"
endif
	@echo ""
	@echo "Targets:"
	@echo "  make all               - Build all test programs"
	@echo "  make tests             - Build all test programs"
	@echo "  make test_model_input_interface - Build specific test"
	@echo "  make clean             - Clean build artifacts"
	@echo "  make rebuild           - Clean and rebuild"
	@echo "  make info              - Show this information"
	@echo "====================================="

# Help target
.PHONY: help
help: info

# ===================================================================
# DEPENDENCIES
# ===================================================================

# Auto-generate dependencies
DEPS := $(LIB_OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d)

-include $(DEPS)

# Dependency generation flags
DEPFLAGS = -MMD -MP -MF $(@:.o=.d)

# Update compilation rules to generate dependencies
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cxx
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cxx
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# ===================================================================
# SPECIAL RULES
# ===================================================================

# Ensure model file exists for testing
.PHONY: check-model
check-model:
	@if [ ! -f "siren_model.pt" ]; then \
		echo "Warning: siren_model.pt not found."; \
		echo "Run 'python make_siren_trace.py config.yaml' to generate it."; \
	fi

# Run the test
.PHONY: run-test
run-test: $(BIN_DIR)/test_model_input_interface check-model
	@echo "Running test_model_input_interface..."
	$(BIN_DIR)/test_model_input_interface siren_model.pt

# Debug build with verbose output
.PHONY: debug
debug: CXXFLAGS += -v
debug: all

.PHONY: verbose
verbose:
	$(MAKE) --debug=v all
