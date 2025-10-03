# Source CMakeLists.txt for flashmatch_dataprep library

# ========================================
# Library Source Files
# ========================================

# Core source files (always built)
set(CORE_SOURCES
    FlashMatchOutputData.cxx
    FlashTrackMatcher.cxx
    CRTMatcher.cxx
    PrepareVoxelOutput.cxx
    PMTPositions.cxx
)

# Core header files
set(CORE_HEADERS
    ${CMAKE_SOURCE_DIR}/include/FlashMatchOutputData.h
    ${CMAKE_SOURCE_DIR}/include/DataStructures.h
    ${CMAKE_SOURCE_DIR}/include/FlashTrackMatcher.h
    ${CMAKE_SOURCE_DIR}/include/CRTMatcher.h
    ${CMAKE_SOURCE_DIR}/include/PrepareVoxelOutput.h
    ${CMAKE_SOURCE_DIR}/include/PMTPositions.h
)

# Optional source files (conditional compilation)
set(OPTIONAL_SOURCES)
set(OPTIONAL_HEADERS)

# Data interface sources (depend on UBDL)
if(larlite_FOUND AND larcv_FOUND)
    list(APPEND OPTIONAL_SOURCES
        LarliteDataInterface.cxx
        CosmicRecoInput.cxx
        TruthFlashTrackMatcher.cxx
    )
    list(APPEND OPTIONAL_HEADERS
        ${CMAKE_SOURCE_DIR}/include/LarliteDataInterface.h
        ${CMAKE_SOURCE_DIR}/include/CosmicRecoInput.h
        ${CMAKE_SOURCE_DIR}/include/TruthFlashTrackMatcher.h
    )
endif()

# HDF5 output source if available
if(HDF5_FOUND AND HighFive_FOUND)
    list(APPEND OPTIONAL_SOURCES FlashMatchHDF5Output.cxx)
    list(APPEND OPTIONAL_HEADERS ${CMAKE_SOURCE_DIR}/include/FlashMatchHDF5Output.h)
endif()

# Torch model interface if available
if(Torch_FOUND)
    list(APPEND OPTIONAL_SOURCES
        SirenTorchModel.cxx
        ModelInputInterface.cxx
        UnbalancedSinkhornDivergence.cxx
        UBFlashSinkDiv.cxx
    )
    list(APPEND OPTIONAL_HEADERS
        ${CMAKE_SOURCE_DIR}/include/SirenTorchModel.h
        ${CMAKE_SOURCE_DIR}/include/ModelInputInterface.h
        ${CMAKE_SOURCE_DIR}/include/UnbalancedSinkhornDivergence.h
        ${CMAKE_SOURCE_DIR}/include/UBFlashSinkDiv.h
    )
endif()

# Combine all sources
set(ALL_SOURCES ${CORE_SOURCES} ${OPTIONAL_SOURCES})
set(ALL_HEADERS ${CORE_HEADERS} ${OPTIONAL_HEADERS})

# ========================================
# Create Shared Library
# ========================================

# Create the shared library
add_library(flashmatch_dataprep SHARED ${ALL_SOURCES})
add_library(flashmatch::dataprep ALIAS flashmatch_dataprep)

# Set library version
set_target_properties(flashmatch_dataprep PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# ========================================
# Target Include Directories
# ========================================

# Public include directories (exported with the library)
target_include_directories(flashmatch_dataprep
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Private include directories (not exported)
target_include_directories(flashmatch_dataprep
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# System dependencies
target_include_directories(flashmatch_dataprep SYSTEM PUBLIC ${ROOT_INCLUDE_DIRS})

# Optional dependencies includes
# Python headers are needed because larflow headers include them
if(Python3_FOUND)
    target_include_directories(flashmatch_dataprep SYSTEM PUBLIC
        ${Python3_INCLUDE_DIRS}
        ${NUMPY_INCLUDES}
    )
endif()

if(HDF5_FOUND)
    target_include_directories(flashmatch_dataprep SYSTEM PUBLIC ${HDF5_INCLUDE_DIRS})
    if(HighFive_FOUND AND DEFINED HIGHFIVE_INCLUDE_DIR)
        # HighFive is header-only, handle it differently for build vs install
        target_include_directories(flashmatch_dataprep SYSTEM PUBLIC
            $<BUILD_INTERFACE:${HIGHFIVE_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include>
        )
        # Install HighFive headers with the library
        install(DIRECTORY ${HIGHFIVE_INCLUDE_DIR}/highfive
            DESTINATION include
        )
    endif()
endif()

if(Torch_FOUND)
    target_include_directories(flashmatch_dataprep SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
endif()

# UBDL includes if available
if(larlite_FOUND)
    target_include_directories(flashmatch_dataprep SYSTEM PUBLIC $ENV{LARLITE_INCDIR})
endif()

if(larcv_FOUND)
    target_include_directories(flashmatch_dataprep SYSTEM PUBLIC $ENV{LARCV_INCDIR})
endif()

if(ublarcvapp_FOUND)
    target_include_directories(flashmatch_dataprep SYSTEM PUBLIC $ENV{UBLARCVAPP_INCDIR})
endif()

if(larflow_FOUND)
    target_include_directories(flashmatch_dataprep SYSTEM PUBLIC $ENV{LARFLOW_INCDIR})
endif()

# ========================================
# Link Libraries
# ========================================

# Public dependencies (users of the library also need these)
target_link_libraries(flashmatch_dataprep
    PUBLIC
        ${ROOT_LIBRARIES}
)

# Python libraries - made PUBLIC because larflow headers expose Python symbols
target_link_libraries(flashmatch_dataprep
    PUBLIC
        ${Python3_LIBRARIES}
)

# Torch libraries if available
if(Torch_FOUND)
    target_link_libraries(flashmatch_dataprep PUBLIC ${TORCH_LIBRARIES})
endif()

# HDF5 libraries if available
if(HDF5_FOUND AND HighFive_FOUND)
    target_link_libraries(flashmatch_dataprep PUBLIC ${HDF5_CXX_LIBRARIES})
    target_compile_definitions(flashmatch_dataprep PRIVATE ${HDF5_DEFINITIONS})
endif()

# Link UBDL libraries if found
if(larlite_FOUND)
    find_library(LARLITE_BASE_LIB larlite_Base HINTS $ENV{LARLITE_LIBDIR})
    find_library(LARLITE_DATAFORMAT_LIB larlite_DataFormat HINTS $ENV{LARLITE_LIBDIR})
    find_library(LARLITE_LARUTIL_LIB larlite_LArUtil HINTS $ENV{LARLITE_LIBDIR})
    if(LARLITE_BASE_LIB AND LARLITE_DATAFORMAT_LIB)
        target_link_libraries(flashmatch_dataprep PUBLIC
            ${LARLITE_BASE_LIB}
            ${LARLITE_DATAFORMAT_LIB}
            ${LARLITE_LARUTIL_LIB}
        )
        target_compile_definitions(flashmatch_dataprep PUBLIC HAVE_LARLITE)
    endif()
endif()

if(larcv_FOUND)
    find_library(LARCV_BASE_LIB LArCVCoreBase HINTS $ENV{LARCV_LIBDIR})
    find_library(LARCV_DATAFORMAT_LIB LArCVCoreDataFormat HINTS $ENV{LARCV_LIBDIR})
    if(LARCV_BASE_LIB AND LARCV_DATAFORMAT_LIB)
        target_link_libraries(flashmatch_dataprep PUBLIC
            ${LARCV_BASE_LIB}
            ${LARCV_DATAFORMAT_LIB}
        )
        target_compile_definitions(flashmatch_dataprep PUBLIC HAVE_LARCV)
    endif()
endif()

if(ublarcvapp_FOUND)
    find_library(UBLARCVAPP_UBDLLEE_LIB LArCVApp_ubdllee HINTS $ENV{UBLARCVAPP_LIBDIR})
    find_library(UBLARCVAPP_IMAGECLUSTER_LIB LArOpenCV_ImageClusterAlgoClass HINTS $ENV{LARLITE_LIBDIR})
    find_library(UBLARCVAPP_MCTOOLS_LIB LArCVApp_MCTools HINTS $ENV{UBLARCVAPP_LIBDIR})
    if(UBLARCVAPP_UBDLLEE_LIB)
        target_link_libraries(flashmatch_dataprep PUBLIC
            ${UBLARCVAPP_UBDLLEE_LIB}
            ${UBLARCVAPP_IMAGECLUSTER_LIB}
            ${UBLARCVAPP_MCTOOLS_LIB}
        )
        target_compile_definitions(flashmatch_dataprep PUBLIC HAVE_UBLARCVAPP)
    endif()
endif()

if(larflow_FOUND)
    find_library(LARFLOW_RECO_LIB LArFlow_Reco HINTS $ENV{LARFLOW_LIBDIR})
    find_library(LARFLOW_RECOUTILS_LIB LArFlow_RecoUtils HINTS $ENV{LARFLOW_LIBDIR})
    find_library(LARFLOW_SCB_LIB LArFlow_SCBoundary HINTS $ENV{LARFLOW_LIBDIR})
    find_library(LARFLOW_VOXELIZER_LIB LArFlow_Voxelizer HINTS $ENV{LARFLOW_LIBDIR})
    find_library(LARFLOW_PREP_LIB LArFlow_PrepFlowMatchData HINTS $ENV{LARFLOW_LIBDIR})
    if(LARFLOW_VOXELIZER_LIB)
        target_link_libraries(flashmatch_dataprep PUBLIC
            ${LARFLOW_VOXELIZER_LIB}
        )
        if(LARFLOW_RECO_LIB AND LARFLOW_RECOUTILS_LIB)
            target_link_libraries(flashmatch_dataprep PUBLIC
                ${LARFLOW_RECO_LIB}
                ${LARFLOW_RECOUTILS_LIB}
                ${LARFLOW_SCB_LIB}
                ${LARFLOW_PREP_LIB}
            )
        endif()
        target_compile_definitions(flashmatch_dataprep PUBLIC HAVE_LARFLOW)
    endif()
endif()

# ========================================
# Compile Definitions
# ========================================

# Export compile definitions for optional features
if(HDF5_FOUND AND HighFive_FOUND)
    target_compile_definitions(flashmatch_dataprep PUBLIC HAVE_HDF5 HAVE_HIGHFIVE)
endif()

if(Torch_FOUND)
    target_compile_definitions(flashmatch_dataprep PUBLIC HAVE_TORCH)
endif()

# ========================================
# Create Executables (Optional)
# ========================================

option(BUILD_FLASHMATCH_APPS "Build flashmatch data preparation executables" ON)

if(BUILD_FLASHMATCH_APPS)
    # Main data preparation executable
    add_executable(flashmatch_dataprep_exe main.cxx)
    set_target_properties(flashmatch_dataprep_exe PROPERTIES OUTPUT_NAME flashmatch_dataprep)
    target_link_libraries(flashmatch_dataprep_exe PRIVATE flashmatch::dataprep)

    # Link Python explicitly for the executable (needed because larflow headers use Python)
    if(Python3_FOUND)
        target_link_libraries(flashmatch_dataprep_exe PRIVATE ${Python3_LIBRARIES})
    endif()

    # MC data preparation executable
    if(larlite_FOUND AND larflow_FOUND)
        add_executable(flashmatch_mcprep make_mc_training_data.cxx)
        target_link_libraries(flashmatch_mcprep PRIVATE flashmatch::dataprep)

        # Link Python explicitly for the executable (needed because larflow headers use Python)
        if(Python3_FOUND)
            target_link_libraries(flashmatch_mcprep PRIVATE ${Python3_LIBRARIES})
        endif()

        # Additional libraries needed for MC prep
        find_library(LARFLOW_OPMODEL_LIB LArFlow_OpticalModel HINTS $ENV{LARFLOW_LIBDIR})
        if(LARFLOW_OPMODEL_LIB)
            target_link_libraries(flashmatch_mcprep PRIVATE ${LARFLOW_OPMODEL_LIB})
        endif()
    endif()

    # Siren Model C++ test executable
    add_executable(test_siren_model_interface test_siren_model_interface.cxx)
    target_link_libraries(test_siren_model_interface PRIVATE flashmatch::dataprep)

    # Sinkhorn Divergence test executable (requires Torch)
    if(Torch_FOUND)
        add_executable(test_sinkhorn test_sinkhorn.cxx)
        target_link_libraries(test_sinkhorn PRIVATE flashmatch::dataprep)

        # Epsilon schedule test
        add_executable(test_epsilon_schedule test_epsilon_schedule.cxx)
        target_link_libraries(test_epsilon_schedule PRIVATE flashmatch::dataprep)
    endif()

endif()

# ========================================
# Installation Rules
# ========================================

# Install the library
install(TARGETS flashmatch_dataprep
    EXPORT flashmatch_dataprepTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install header files
install(FILES ${ALL_HEADERS}
    DESTINATION include/flashmatch_dataprep
)

# Install executables if built
if(BUILD_FLASHMATCH_APPS)
    install(TARGETS flashmatch_dataprep_exe
        RUNTIME DESTINATION bin
    )
    if(TARGET flashmatch_mcprep)
        install(TARGETS flashmatch_mcprep
            RUNTIME DESTINATION bin
        )
    endif()

    install(TARGETS test_siren_model_interface
        RUNTIME DESTINATION bin
    )

    if(TARGET test_sinkhorn)
        install(TARGETS test_sinkhorn
            RUNTIME DESTINATION bin
        )
    endif()

    if(TARGET test_epsilon_schedule)
        install(TARGETS test_epsilon_schedule
            RUNTIME DESTINATION bin
        )
    endif()
endif()

# Export targets
install(EXPORT flashmatch_dataprepTargets
    FILE flashmatch_dataprepTargets.cmake
    NAMESPACE flashmatch::
    DESTINATION lib/cmake/flashmatch_dataprep
)

# ========================================
# Generate Config Files
# ========================================

# This is handled in the parent CMakeLists.txt
