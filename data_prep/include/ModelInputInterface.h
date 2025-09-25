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
#include "larflow/Voxelizer/VoxelChargeCalculator.h"

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

    void prepare_input_tensor(
        const larflow::voxelizer::VoxelChargeCalculator::VoxelChargeInfo_t& voxelchargeinfo,
        torch::Tensor& voxel_features,
        torch::Tensor& voxel_charge,
        torch::Tensor& mask );

    // Setters for normalization parameters
    void set_planecharge_normalization(const std::vector<float>& offsets, const std::vector<float>& scales) {
        _planecharge_offset = offsets;
        _planecharge_scale = scales;
    }

    void set_pmt_normalization(float offset, float scale) {
        _pmt_offset = offset;
        _pmt_scale = scale;
    }

    void set_use_log_normalization(bool use_log) {
        _use_log_normalization = use_log;
    }

    // Getter for PMT positions if needed externally
    torch::Tensor get_pmt_positions() const { return _pmtpos; }

protected:

    void _define_voxels();
    void _define_pmt_positions();
    torch::Tensor _createPMTPositionsTensor();

    // Helper functions for input preparation
    torch::Tensor _normalize_planecharge(const torch::Tensor& planecharge);

    void _calc_dist_to_pmts(
        const torch::Tensor& pos_cm,
        torch::Tensor& dist2pmts,
        torch::Tensor& dvec2pmts);

    void _prepare_mlp_input_variables(
        const torch::Tensor& coord,
        const torch::Tensor& q_perplane,
        torch::Tensor& voxel_features,
        torch::Tensor& voxel_charge);

    // Member variables
    float _voxel_len_cm;
    larflow::voxelizer::VoxelizeTriplets     _voxelizer; ///< defines voxels, assigns points to voxels
    flashmatch::dataprep::PrepareVoxelOutput _assign_voxel_charge; ///< uses points and assigns charge to voxels

    // PMT positions tensor (32, 3)
    torch::Tensor _pmtpos;

    // Normalization parameters
    bool _use_log_normalization;
    std::vector<float> _planecharge_offset;
    std::vector<float> _planecharge_scale;
    float _pmt_offset;
    float _pmt_scale;


};


}

#endif
#endif
#endif