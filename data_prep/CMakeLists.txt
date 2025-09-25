cmake_minimum_required(VERSION 3.10)
project(flashmatch_dataprep VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

#---Install location by default is in the build directory
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/installed" CACHE PATH "${CMAKE_BUILD_DIR}" FORCE )
endif()

#---Module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

## Dependencies

#---CERN ROOT
find_package(ROOT CONFIG REQUIRED)
# Set compiler version based on ROOT
execute_process(
    COMMAND bash "-c" "root-config --features"
    OUTPUT_VARIABLE TEST_ROOT_CXX_STANDARD )
if (${TEST_ROOT_CXX_STANDARD} MATCHES "cxx17")
  message( STATUS "using cxx17")
  set(CMAKE_CXX_STANDARD 17)
elseif (${TEST_ROOT_CXX_STANDARD} MATCHES "cxx14")
  message( STATUS "using cxx14")
  set(CMAKE_CXX_STANDARD 14)
else ()
  message( STATUS "using cxx11")  
  set(CMAKE_CXX_STANDARD 11)
endif()

#---Python for integration with Python visualization tools
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process( COMMAND python3 -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDES OUTPUT_STRIP_TRAILING_WHITESPACE ) 

# ========================
# UBDL MODULES (OPTIONAL)
# ========================

#---larlite
if(DEFINED ENV{LARLITE_LIBDIR})
    set( larlite_DIR "$ENV{LARLITE_LIBDIR}/cmake/larlite" )
    find_package( larlite QUIET )
    if(larlite_FOUND)
        message(STATUS "Found larlite")
    else()
        message(WARNING "larlite not found, some features will be disabled")
    endif()
else()
    message(STATUS "LARLITE_LIBDIR not set, skipping larlite")
endif()

#---larcv
if(DEFINED ENV{LARCV_LIBDIR})
    set( larcv_DIR "$ENV{LARCV_LIBDIR}/cmake/larcv")
    find_package( larcv QUIET )
    if(larcv_FOUND)
        message(STATUS "Found larcv")
    else()
        message(WARNING "larcv not found, some features will be disabled")
    endif()
else()
    message(STATUS "LARCV_LIBDIR not set, skipping larcv")
endif()

#---ublarcvapp
if(DEFINED ENV{UBLARCVAPP_LIBDIR})
    set( ublarcvapp_DIR "$ENV{UBLARCVAPP_LIBDIR}/cmake/ublarcvapp")
    find_package( ublarcvapp QUIET )
    if(ublarcvapp_FOUND)
        message(STATUS "Found ublarcvapp")
    else()
        message(WARNING "ublarcvapp not found, some features will be disabled")
    endif()
else()
    message(STATUS "UBLARCVAPP_LIBDIR not set, skipping ublarcvapp")
endif()

#---larflow
if(DEFINED ENV{LARFLOW_LIBDIR})
    set( larflow_DIR "$ENV{LARFLOW_LIBDIR}/cmake/larflow")
    find_package( larflow QUIET )
    if(larflow_FOUND)
        message(STATUS "Found larflow")
    else()
        message(WARNING "larflow not found, some features will be disabled")
    endif()
else()
    message(STATUS "LARFLOW_LIBDIR not set, skipping larflow")
endif()

## Optional Dependencies

#---Eigen3 (for geometric computations)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
    add_compile_definitions(HAVE_EIGEN3)
endif()

#---HDF5 (for machine learning data output)
find_package(HDF5 COMPONENTS C CXX QUIET)
if(HDF5_FOUND)
    message(STATUS "Found HDF5: ${HDF5_VERSION}")
    add_compile_definitions(HAVE_HDF5)
else()
    message(STATUS "HDF5 not found, HDF5 output features will be disabled")
endif()

#---HighFive (header-only HDF5 C++ interface)
if(HDF5_FOUND)
    # Try to find HighFive as an installed package first
    find_package(HighFive QUIET)
    if(NOT HighFive_FOUND)
        # If not found, check if it exists as a submodule or in a known location
        if(EXISTS "${CMAKE_SOURCE_DIR}/extern/highfive/include")
            set(HIGHFIVE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/extern/highfive/include")
            message(STATUS "Found HighFive headers at: ${HIGHFIVE_INCLUDE_DIR}")
            set(HighFive_FOUND TRUE)
        else()
            message(STATUS "HighFive not found. To use HDF5 features, please:")
            message(STATUS "  1. Install HighFive: https://github.com/highfive-devs/highfive/tree/main ")
            message(STATUS "  2. Or add as submodule: git submodule add https://github.com/highfive-devs/highfive/tree/main extern/highfive")
        endif()
    else()
        message(STATUS "Found HighFive package")
    endif()
    
    if(HighFive_FOUND)
        add_compile_definitions(HAVE_HIGHFIVE)
    endif()
endif()

# ======================================================
# TORCH DEPENDENCIES FOR C++ INTERFACE

set( Torch_DIR "$ENV{LIBTORCH_DIR}/share/cmake/Torch" )
find_package( Torch REQUIRED )

# Now check LibTorch libraries for ABI compatibility
# Look for the torch library file - check multiple possible locations
set(TORCH_LIB_LOCATION "")
set(POSSIBLE_TORCH_LIBS 
    "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so"
    "${TORCH_LIBRARIES}"
    "${CMAKE_PREFIX_PATH}/lib/libtorch_python.so"
    "$ENV{LIBTORCH_DIR}/lib/libtorch_python.so"
)

foreach(LIB ${POSSIBLE_TORCH_LIBS})
    if(EXISTS ${LIB})
        set(TORCH_LIB_LOCATION ${LIB})
        break()
    endif()
endforeach()

if(NOT EXISTS ${TORCH_LIB_LOCATION})
    # Try to get it from the torch target
    get_target_property(TORCH_LIB_LOCATION torch LOCATION)
endif()

message(STATUS "Checking LibTorch library at: ${TORCH_LIB_LOCATION}")

# Use strings command to check for CXX11 ABI symbols
find_program(STRINGS_EXECUTABLE strings)

set(TORCH_ABI_DETECTED "unknown")

if(STRINGS_EXECUTABLE AND EXISTS ${TORCH_LIB_LOCATION})
    # Look for cxx1112basic_string pattern which indicates CXX11 ABI
    execute_process(
        COMMAND ${STRINGS_EXECUTABLE} ${TORCH_LIB_LOCATION}
        COMMAND grep "cxx1112basic_string"
        COMMAND head -n 1
        OUTPUT_VARIABLE STRINGS_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(STRINGS_OUTPUT)
        set(TORCH_ABI_DETECTED "1")
        message(STATUS "Detected LibTorch was built with CXX11 ABI (found cxx1112basic_string symbols)")
    else()
        # Double check by looking for the __cxx11 namespace
        execute_process(
            COMMAND ${STRINGS_EXECUTABLE} ${TORCH_LIB_LOCATION}
            COMMAND grep "__cxx11"
            COMMAND head -n 1
            OUTPUT_VARIABLE STRINGS_OUTPUT2
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(STRINGS_OUTPUT2)
            set(TORCH_ABI_DETECTED "1")
            message(STATUS "Detected LibTorch was built with CXX11 ABI (found __cxx11 namespace)")
        else()
            set(TORCH_ABI_DETECTED "0")
            message(STATUS "LibTorch appears to be built with old ABI (no CXX11 markers found)")
        endif()
    endif()
else()
    message(WARNING "Could not find 'strings' command to check LibTorch ABI. Build may fail if ABI is mismatched.")
endif()

# Check if ABIs match
if(COMPILER_ABI_VALUE STREQUAL "1" AND TORCH_ABI_DETECTED STREQUAL "0")
    message(FATAL_ERROR "\n\nLibTorch ABI Compatibility Error:\n"
        "Your compiler/dependencies use _GLIBCXX_USE_CXX11_ABI=1,\n"
        "but LibTorch was built with _GLIBCXX_USE_CXX11_ABI=0.\n\n"
        "Please install a CXX11 ABI version of LibTorch.\n"
        "You can download pre-compiled versions from:\n"
        "https://download.pytorch.org/libtorch/cu111 (or appropriate CUDA version)\n"
        "Look for versions with 'cxx11-abi' in the filename.\n\n"
        "Set the environment variable LIBTORCH_DIR to point to the CXX11 ABI version.")
elseif(COMPILER_ABI_VALUE STREQUAL "0" AND TORCH_ABI_DETECTED STREQUAL "1")
    message(FATAL_ERROR "\n\nLibTorch ABI Compatibility Error:\n"
        "Your compiler/dependencies use _GLIBCXX_USE_CXX11_ABI=0,\n"
        "but LibTorch was built with _GLIBCXX_USE_CXX11_ABI=1.\n\n"
        "You need to either:\n"
        "1. Install a non-CXX11 ABI version of LibTorch, or\n"
        "2. Rebuild your dependencies with CXX11 ABI enabled.")
elseif(TORCH_ABI_DETECTED STREQUAL "unknown")
    message(WARNING "\n\nLibTorch ABI Compatibility Warning:\n"
        "Could not detect LibTorch ABI version.\n"
        "Build may fail if there is an ABI mismatch.\n\n"
        "If you encounter linking errors, ensure LibTorch matches your compiler's ABI:\n"
        "Your compiler uses _GLIBCXX_USE_CXX11_ABI=${COMPILER_ABI_VALUE}")
else()
    message(STATUS "LibTorch ABI compatibility check passed (both using CXX11_ABI=${COMPILER_ABI_VALUE})")
endif()

# END OF TORCH SECTION
# ======================================================

## Setup

#---Define useful ROOT functions and macros
include(${ROOT_USE_FILE})

#---Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-deprecated-copy -Wno-ignored-qualifiers -Wno-reorder")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set RPATH for installed libraries
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

## Add Source

#---Add base of source
add_subdirectory(src)

# Option to build example applications
option(BUILD_EXAMPLES "Build example applications" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

#---Configuration export
include(CMakePackageConfigHelpers)

# Set variables for the config file
set(FLASHMATCH_DATAPREP_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(FLASHMATCH_DATAPREP_LIBRARY_DIR "${CMAKE_INSTALL_PREFIX}/lib")

# Generate the config file
configure_package_config_file(
  "cmake/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/flashmatch_dataprepConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/flashmatch_dataprep"
  PATH_VARS FLASHMATCH_DATAPREP_INCLUDE_DIR FLASHMATCH_DATAPREP_LIBRARY_DIR
)

# Generate the version file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/flashmatch_dataprepConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

## Export and Installation

# Install the generated config files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/flashmatch_dataprepConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/flashmatch_dataprepConfigVersion.cmake"
  DESTINATION "lib/cmake/flashmatch_dataprep"
)

# Create pkg-config file for non-CMake users
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/flashmatch_dataprep.pc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/flashmatch_dataprep.pc"
  @ONLY
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/flashmatch_dataprep.pc"
  DESTINATION "lib/pkgconfig"
)
