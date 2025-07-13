cmake_minimum_required(VERSION 3.10)
project(flashmatch_dataprep VERSION 1.0.0)

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

## Setup

#---Define useful ROOT functions and macros
include(${ROOT_USE_FILE})

#---Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

## Add Source

#---Add base of source
add_subdirectory(src)

#---Configuration export
include(CMakePackageConfigHelpers)
set(FLASHMATCH_DATAPREP_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(FLASHMATCH_DATAPREP_LIBRARY_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(TARGETS_EXPORT_NAME "flashmatch_dataprepTargets")
configure_package_config_file("cmake/Config.cmake.in"
  "${CMAKE_BINARY_DIR}/cmake/flashmatch_dataprepConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/flashmatch_dataprep"
  PATH_VARS FLASHMATCH_DATAPREP_INCLUDE_DIR FLASHMATCH_DATAPREP_LIBRARY_DIR)
write_basic_package_version_file("${CMAKE_BINARY_DIR}/cmake/flashmatch_dataprepConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion)

## Export Targets

#---Create info to include flashmatch_dataprep in other cmake files
install(EXPORT flashmatch_dataprep
        FILE flashmatch_dataprepTargets.cmake
        DESTINATION "${CMAKE_BINARY_DIR}/installed/lib/cmake/flashmatch_dataprep")

#---Create info to include flashmatch_dataprep in other cmake files
install(FILES "${CMAKE_BINARY_DIR}/cmake/flashmatch_dataprepConfig.cmake"
  "${CMAKE_BINARY_DIR}/cmake/flashmatch_dataprepConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/flashmatch_dataprep")