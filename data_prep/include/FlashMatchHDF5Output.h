#ifndef __FLASHMATCH_DATAPREP_FLASHMATCHHDF5OUTPUT_H__
#define __FLASHMATCH_DATAPREP_FLASHMATCHHDF5OUTPUT_H__

#include <string>
#include <vector>
#include <memory>
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5Easy.hpp>

#include "DataStructures.h"

namespace flashmatch {
namespace dataprep {

/**
 * @brief Stores voxel data from flash matching to HDF5 format using HighFive
 * 
 * This class provides functionality to save the voxel-related branches from
 * the ROOT output into HDF5 format for easier integration with machine learning
 * frameworks.
 */
class FlashMatchHDF5Output {
public:
    /**
     * @brief Constructor
     * @param output_h5file Path to the output HDF5 file
     * @param allow_overwrite If true, overwrites existing file; if false, appends
     */
    FlashMatchHDF5Output(const std::string& output_h5file, bool allow_overwrite = true);
    
    /**
     * @brief Destructor - ensures file is properly closed
     */
    ~FlashMatchHDF5Output();
    
    /**
     * @brief Store voxel data for a single match
     * @param matched_data EventData containing all matched information
     * @param match_index Index of the specific match to store
     * @return True if successful, false otherwise
     */
    bool storeVoxelData(const EventData& matched_data, int match_index);
    
    /**
     * @brief Store all voxel data for an entire event
     * @param matched_data EventData containing all matched information
     * @return Number of matches stored
     */
    int storeEventVoxelData(const EventData& matched_data);
    
    /**
     * @brief Clear internal buffers
     */
    void clear();
    
    /**
     * @brief Flush data to disk
     */
    void flush();
    
    /**
     * @brief Close the HDF5 file
     */
    void close();
    
    /**
     * @brief Check if file is open
     */
    bool isOpen() const;

private:
    std::unique_ptr<HighFive::File> _h5file;
    
    // Group names for organization
    const std::string VOXEL_GROUP = "/voxel_data";
    const std::string EVENT_GROUP = "/event_info";
    
    // Dataset names
    const std::string VOXEL_PLANECHARGE_DS = "voxel_planecharge";
    const std::string VOXEL_INDICES_DS = "voxel_indices";
    const std::string VOXEL_AVEPOS_DS = "voxel_avepos";
    const std::string VOXEL_CENTERS_DS = "voxel_centers";
    const std::string EVENT_INFO_DS = "event_info";
    
    // Internal buffers for batching writes
    struct VoxelBatch {
        std::vector<std::vector<std::vector<float>>> planecharge_batch;
        std::vector<std::vector<std::vector<int>>> indices_batch;
        std::vector<std::vector<std::vector<float>>> avepos_batch;
        std::vector<std::vector<std::vector<float>>> centers_batch;
        std::vector<int> run_batch;
        std::vector<int> subrun_batch;
        std::vector<int> event_batch;
        std::vector<int> match_index_batch;
        
        void clear() {
            planecharge_batch.clear();
            indices_batch.clear();
            avepos_batch.clear();
            centers_batch.clear();
            run_batch.clear();
            subrun_batch.clear();
            event_batch.clear();
            match_index_batch.clear();
        }
        
        size_t size() const {
            return run_batch.size();
        }
    };
    
    VoxelBatch _batch;
    size_t _batch_size;
    size_t _total_entries;
    
    /**
     * @brief Initialize HDF5 file structure
     */
    void initializeFile();
    
    /**
     * @brief Write current batch to file
     */
    void writeBatch();
    
    /**
     * @brief Create or extend a dataset for nested vectors of floats
     */
    void writeNestedFloatVector(const std::string& group_name,
                                const std::string& dataset_name,
                                const std::vector<std::vector<std::vector<float>>>& data);
    
    /**
     * @brief Create or extend a dataset for nested vectors of ints
     */
    void writeNestedIntVector(const std::string& group_name,
                             const std::string& dataset_name,
                             const std::vector<std::vector<std::vector<int>>>& data);
};

} // namespace dataprep
} // namespace flashmatch

#endif // __FLASHMATCH_DATAPREP_FLASHMATCHHDF5OUTPUT_H__