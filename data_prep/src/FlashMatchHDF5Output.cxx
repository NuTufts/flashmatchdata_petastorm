#include "FlashMatchHDF5Output.h"
#include <iostream>
#include <stdexcept>

namespace flashmatch {
namespace dataprep {

FlashMatchHDF5Output::FlashMatchHDF5Output(const std::string& output_h5file, bool allow_overwrite)
    : _batch_size(100), _total_entries(0)
{
    try {
        HighFive::File::AccessMode flags = HighFive::File::ReadWrite | HighFive::File::Create;
        if (allow_overwrite) {
            flags = flags | HighFive::File::Truncate;
        } else {
            flags = flags | HighFive::File::Excl;
        }
        
        _h5file = std::make_unique<HighFive::File>(output_h5file, flags);
        initializeFile();
        
    } catch (const HighFive::Exception& e) {
        throw std::runtime_error("Failed to create/open HDF5 file: " + output_h5file + 
                               "\nHighFive error: " + e.what());
    }
}

FlashMatchHDF5Output::~FlashMatchHDF5Output()
{
    if (_h5file && _batch.size() > 0) {
        writeBatch();
    }
    close();
}

void FlashMatchHDF5Output::initializeFile()
{
    if (!_h5file) return;
    
    // Create main groups
    if (!_h5file->exist(VOXEL_GROUP)) {
        _h5file->createGroup(VOXEL_GROUP);
    }
    if (!_h5file->exist(EVENT_GROUP)) {
        _h5file->createGroup(EVENT_GROUP);
    }
    
    // Add attributes to describe the data
    auto voxel_group = _h5file->getGroup(VOXEL_GROUP);
    voxel_group.createAttribute<std::string>("description", 
        HighFive::DataSpace::From("Voxel data from flash matching"));
    voxel_group.createAttribute<std::string>("format_version", 
        HighFive::DataSpace::From("1.0"));
}

bool FlashMatchHDF5Output::storeVoxelData(const EventData& matched_data, int match_index)
{
    if (!_h5file) {
        std::cerr << "HDF5 file not open" << std::endl;
        return false;
    }
    
    if (match_index < 0 || match_index >= (int)matched_data.voxel_planecharge_vvv.size()) {
        std::cerr << "Invalid match index: " << match_index << std::endl;
        return false;
    }
    
    // Add to batch
    _batch.planecharge_batch.push_back(matched_data.voxel_planecharge_vvv[match_index]);
    _batch.indices_batch.push_back(matched_data.voxel_indices_vvv[match_index]);
    _batch.avepos_batch.push_back(matched_data.voxel_avepos_vvv[match_index]);
    _batch.centers_batch.push_back(matched_data.voxel_centers_vvv[match_index]);
    
    _batch.run_batch.push_back(matched_data.run);
    _batch.subrun_batch.push_back(matched_data.subrun);
    _batch.event_batch.push_back(matched_data.event);
    _batch.match_index_batch.push_back(match_index);
    
    // Write batch if it reaches the batch size
    if (_batch.size() >= _batch_size) {
        writeBatch();
    }
    
    return true;
}

int FlashMatchHDF5Output::storeEventVoxelData(const EventData& matched_data)
{
    int n_matches = matched_data.cosmic_tracks.size();
    int n_stored = 0;
    
    for (int i = 0; i < n_matches; i++) {
        if (storeVoxelData(matched_data, i)) {
            n_stored++;
        }
    }
    
    return n_stored;
}

void FlashMatchHDF5Output::writeBatch()
{
    if (!_h5file || _batch.size() == 0) return;
    
    try {
        // For simplicity in this implementation, we'll write each entry as a separate dataset
        // In a production system, you might want to use a more sophisticated schema
        
        auto voxel_group = _h5file->getGroup(VOXEL_GROUP);
        auto event_group = _h5file->getGroup(EVENT_GROUP);
        
        for (size_t i = 0; i < _batch.size(); i++) {
            size_t entry_idx = _total_entries + i;
            std::string entry_name = "entry_" + std::to_string(entry_idx);
            
            // Create a subgroup for this entry
            auto entry_group = voxel_group.createGroup(entry_name);
            
            // Write voxel data using H5Easy for simplicity
            // Note: H5Easy handles nested vectors well
            H5Easy::dump(*_h5file, VOXEL_GROUP + "/" + entry_name + "/planecharge", 
                        _batch.planecharge_batch[i]);
            H5Easy::dump(*_h5file, VOXEL_GROUP + "/" + entry_name + "/indices", 
                        _batch.indices_batch[i]);
            H5Easy::dump(*_h5file, VOXEL_GROUP + "/" + entry_name + "/avepos", 
                        _batch.avepos_batch[i]);
            H5Easy::dump(*_h5file, VOXEL_GROUP + "/" + entry_name + "/centers", 
                        _batch.centers_batch[i]);
            
            // Write event info as attributes
            entry_group.createAttribute<int>("run", HighFive::DataSpace::From(_batch.run_batch[i]));
            entry_group.createAttribute<int>("subrun", HighFive::DataSpace::From(_batch.subrun_batch[i]));
            entry_group.createAttribute<int>("event", HighFive::DataSpace::From(_batch.event_batch[i]));
            entry_group.createAttribute<int>("match_index", HighFive::DataSpace::From(_batch.match_index_batch[i]));
        }
        
        // Alternative: Write as a single large dataset with compound type
        // This would be more efficient for large datasets but requires more complex handling
        // of variable-length nested vectors
        
        _total_entries += _batch.size();
        _batch.clear();
        
    } catch (const HighFive::Exception& e) {
        std::cerr << "Error writing batch to HDF5: " << e.what() << std::endl;
        throw;
    }
}

void FlashMatchHDF5Output::writeNestedFloatVector(const std::string& group_name,
                                                  const std::string& dataset_name,
                                                  const std::vector<std::vector<std::vector<float>>>& data)
{
    // This is a helper function for writing nested vectors
    // HighFive and H5Easy handle this automatically, but this shows how you could
    // do it manually if needed
    
    // For variable-length nested data, we typically flatten it and store dimensions separately
    // Or use H5Easy which handles this automatically
    H5Easy::dump(*_h5file, group_name + "/" + dataset_name, data);
}

void FlashMatchHDF5Output::writeNestedIntVector(const std::string& group_name,
                                               const std::string& dataset_name,
                                               const std::vector<std::vector<std::vector<int>>>& data)
{
    H5Easy::dump(*_h5file, group_name + "/" + dataset_name, data);
}

void FlashMatchHDF5Output::clear()
{
    _batch.clear();
}

void FlashMatchHDF5Output::flush()
{
    if (_batch.size() > 0) {
        writeBatch();
    }
    if (_h5file) {
        _h5file->flush();
    }
}

void FlashMatchHDF5Output::close()
{
    if (_h5file) {
        if (_batch.size() > 0) {
            writeBatch();
        }
        _h5file.reset();
    }
}

bool FlashMatchHDF5Output::isOpen() const
{
    return _h5file != nullptr;
}

} // namespace dataprep
} // namespace flashmatch