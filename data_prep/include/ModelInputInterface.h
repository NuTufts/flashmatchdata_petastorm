#ifndef __FLASHMATCH_DATAPREP_MODEL_INPUT_INTERFACE__
#define __FLASHMATCH_DATAPREP_MODEL_INPUT_INTERFACE__

/**
 * @brief Given larflow outputs, convert data into tensors for models
 * 
 */

// Prevent the ROOT Interpretter from parsing anything in this header
#ifndef __CINT__
#ifndef __CLING__

#include <vector>
#include <torch/torch.h>

#include "larcv/core/DataFormat/Image2D.h"
#include "larflow/Voxelizer/VoxelizeTriplets.h"

#include "PrepareVoxelOutput.h"

namespace flashmatch {

class ModelInputInterface {

public:

    ModelInputInterface();
    ~ModelInputInterface() {};

    void prepare_input_tensor( 
        const std::vector< std::vector<float> >& hitpos_v, 
        const std::vector< std::vector<float> >& hitimgcoord_v,
        const std::vector< larcv::Image2D >& wireplane_v,
        torch::Tensor& voxel_features,
        torch::Tensor& voxel_charge );


protected:

    void _define_voxels();

    float _voxel_len_cm;
    larflow::voxelizer::VoxelizeTriplets     _voxelizer; ///< defines voxels, assigns points to voxels
    flashmatch::dataprep::PrepareVoxelOutput _assign_voxel_charge; ///< uses points and assigns charge to voxels


};


}

#endif
#endif
#endif