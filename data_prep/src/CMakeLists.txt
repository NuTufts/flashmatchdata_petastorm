# Source CMakeLists.txt for flashmatch_dataprep

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ROOT_INCLUDE_DIRS})

# Include ubdl directories if found
if(larlite_FOUND)
    include_directories($ENV{LARLITE_INCDIR})
endif()

if(larcv_FOUND)
    include_directories($ENV{LARCV_INCDIR})
endif()

if(ublarcvapp_FOUND)
    include_directories($ENV{UBLARCVAPP_INCDIR})
endif()

if(larflow_FOUND)
    include_directories($ENV{LARFLOW_INCDIR})
endif()

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})

# Include HDF5 and HighFive if available
if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    if(HighFive_FOUND)
        if(DEFINED HIGHFIVE_INCLUDE_DIR)
            include_directories(${HIGHFIVE_INCLUDE_DIR})
        endif()
    endif()
endif()

# Collect source files
set(SOURCES
    FlashMatchOutputData.cxx
    FlashTrackMatcher.cxx
    CRTMatcher.cxx
    LarliteDataInterface.cxx
    CosmicRecoInput.cxx
    PrepareVoxelOutput.cxx
    main.cxx
)

# Add HDF5 output source if available
if(HDF5_FOUND AND HighFive_FOUND)
    list(APPEND SOURCES FlashMatchHDF5Output.cxx)
endif()

# Collect header files for installation
set(HEADERS
    ${CMAKE_SOURCE_DIR}/include/FlashMatchOutputData.h
    ${CMAKE_SOURCE_DIR}/include/DataStructures.h
    ${CMAKE_SOURCE_DIR}/include/FlashTrackMatcher.h
    ${CMAKE_SOURCE_DIR}/include/CRTMatcher.h
    ${CMAKE_SOURCE_DIR}/include/LarliteDataInterface.h
    ${CMAKE_SOURCE_DIR}/include/CosmicRecoInput.h
    ${CMAKE_SOURCE_DIR}/include/PrepareVoxelOutput.h
)

# Add HDF5 header if available
if(HDF5_FOUND AND HighFive_FOUND)
    list(APPEND HEADERS ${CMAKE_SOURCE_DIR}/include/FlashMatchHDF5Output.h)
endif()

# Create the main executable
add_executable(flashmatch_dataprep ${SOURCES})

# Link libraries
target_link_libraries(flashmatch_dataprep
    ${ROOT_LIBRARIES}
    ${Python3_LIBRARY}
)

# Link HDF5 libraries if available
if(HDF5_FOUND AND HighFive_FOUND)
    target_link_libraries(flashmatch_dataprep ${HDF5_CXX_LIBRARIES})
    target_compile_definitions(flashmatch_dataprep PRIVATE ${HDF5_DEFINITIONS})
endif()

# Link ubdl libraries if found - using explicit library paths for now
# TODO: Fix target names once proper CMake exports are available
if(larlite_FOUND)
    # Try linking with explicit library files
    find_library(LARLITE_BASE_LIB larlite_Base HINTS $ENV{LARLITE_LIBDIR})
    find_library(LARLITE_DATAFORMAT_LIB larlite_DataFormat HINTS $ENV{LARLITE_LIBDIR})
    find_library(LARLITE_LARUTIL_LIB larlite_LArUtil HINTS $ENV{LARLITE_LIBDIR})
    if(LARLITE_BASE_LIB AND LARLITE_DATAFORMAT_LIB)
        target_link_libraries(flashmatch_dataprep ${LARLITE_BASE_LIB} ${LARLITE_DATAFORMAT_LIB} ${LARLITE_LARUTIL_LIB})
        target_compile_definitions(flashmatch_dataprep PRIVATE HAVE_LARLITE)
        message(STATUS "Linked larlite libraries")
    else()
        message(WARNING "larlite libraries not found, continuing without them")
    endif()
endif()

if(larcv_FOUND)
    find_library(LARCV_BASE_LIB LArCVCoreBase HINTS $ENV{LARCV_LIBDIR})
    find_library(LARCV_DATAFORMAT_LIB LArCVCoreDataFormat HINTS $ENV{LARCV_LIBDIR})
    if(LARCV_BASE_LIB AND LARCV_DATAFORMAT_LIB)
        target_link_libraries(flashmatch_dataprep ${LARCV_BASE_LIB} ${LARCV_DATAFORMAT_LIB})
        target_compile_definitions(flashmatch_dataprep PRIVATE HAVE_LARCV)
        message(STATUS "Linked larcv libraries")
    else()
        message(WARNING "larcv libraries not found, continuing without them")
    endif()
endif()

if(ublarcvapp_FOUND)
    find_library(UBLARCVAPP_UBDLLEE_LIB  LArCVApp_ubdllee HINTS $ENV{UBLARCVAPP_LIBDIR})
    find_library(UBLARCVAPP_IMAGECLUSTER_LIB  LArOpenCV_ImageClusterAlgoClass  HINTS $ENV{LARLITE_LIBDIR})
    if(UBLARCVAPP_UBDLLEE_LIB)
        target_link_libraries(flashmatch_dataprep ${UBLARCVAPP_UBDLLEE_LIB} ${UBLARCVAPP_IMAGECLUSTER_LIB} )
        target_compile_definitions(flashmatch_dataprep PRIVATE HAVE_UBLARCVAPP)
        message(STATUS "Linked ublarcvapp libraries")
    else()
        message(WARNING "ublarcvapp libraries not found, continuing without them")
    endif()
endif()

if(larflow_FOUND)
    find_library(LARFLOW_RECO_LIB LArFlow_Reco HINTS $ENV{LARFLOW_LIBDIR})
    find_library(LARFLOW_RECOUTILS_LIB LArFlow_RecoUtils HINTS $ENV{LARFLOW_LIBDIR})
    find_library(LARFLOW_SCB_LIB LArFlow_SCBoundary HINTS $ENV{LARFLOW_LIBDIR})
    find_library(LARFLOW_VOXELIZER_LIB LArFlow_Voxelizer HINTS $ENV{LARFLOW_LIBDIR})
    find_library(LARFLOW_PREP_LIB LArFlow_PrepFlowMatchData HINTS $ENV{LARFLOW_LIBDIR})
    if(LARFLOW_RECO_LIB AND LARFLOW_RECOUTILS_LIB)
        target_link_libraries(flashmatch_dataprep ${LARFLOW_RECO_LIB} 
            ${LARFLOW_RECOUTILS_LIB} 
            ${LARFLOW_SCB_LIB}
            ${LARFLOW_PREP_LIB}
            ${LARFLOW_VOXELIZER_LIB} )
        target_compile_definitions(flashmatch_dataprep PRIVATE HAVE_LARFLOW)
        message(STATUS "Linked larflow libraries")
    else()
        message(WARNING "larflow libraries not found, continuing without them")
    endif()
endif()

# Set compile features (match ROOT's C++ standard)
target_compile_features(flashmatch_dataprep PRIVATE cxx_std_14)

# Set properties
set_target_properties(flashmatch_dataprep PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install the executable
install(TARGETS flashmatch_dataprep
    RUNTIME DESTINATION bin
)

# Install headers
install(FILES ${HEADERS}
    DESTINATION include/flashmatch_dataprep
)

# Add to export set
install(TARGETS flashmatch_dataprep
    EXPORT flashmatch_dataprep
    RUNTIME DESTINATION bin
)
