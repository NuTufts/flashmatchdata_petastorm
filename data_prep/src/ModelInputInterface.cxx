#include "ModelInputInterface.h"

#include "DataStructures.h"
#include "PMTPositions.h"
#include <cmath>
#include <iostream>

namespace flashmatch {

ModelInputInterface::ModelInputInterface()
{
    _voxel_len_cm = 5.0;
    _define_voxels();
    _define_pmt_positions();

    // Set normalization parameters (from config)
    _use_log_normalization = false;  // Set to true if using log transform

    // Plane charge normalization parameters (linear transform)
    _planecharge_offset = {0.0, 0.0, 0.0};
    _planecharge_scale = {50000.0, 50000.0, 50000.0};

    // PMT normalization parameters (if needed for output)
    _pmt_offset = 0.0;
    _pmt_scale = 5000.0;
}

void ModelInputInterface::_define_voxels()
{
    // set default voxel grid definition
    _voxelizer.set_voxel_size_cm( _voxel_len_cm ); // re-define voxels to 5 cm spaces
    auto const ndims_v = _voxelizer.get_dim_len(); // number of voxels per dimension
    auto const voxel_origin_v = _voxelizer.get_origin(); // (x,y,z) of origin voxel (0,0,0)
    std::vector<float> tpc_origin = { 0.0, -117.0, 0.0 };
    std::vector<float> tpc_end = { 256.0, 117.0, 1036.0 };
    std::vector<int> index_tpc_origin(3,0);
    std::vector<int> index_tpc_end(3,0);
    for (int i=0; i<3; i++) {
        index_tpc_origin[i] = _voxelizer.get_axis_voxel(i,tpc_origin[i]);
        index_tpc_end[i]    = _voxelizer.get_axis_voxel(i,tpc_end[i]);
    }

    std::cout << "VOXELIZER SETUP FOR ModelInputInterface =====================" << std::endl;
    std::cout << "origin: (" << tpc_origin[0] << "," << tpc_origin[1] << "," << tpc_origin[2] << ")" << std::endl;
    std::cout << "ndims: (" << ndims_v[0] << "," << ndims_v[1] << "," << ndims_v[2] << ")" << std::endl;
    std::cout << "index-tpc-origin: ("
              << index_tpc_origin[0] << ","
              << index_tpc_origin[1] << ","
              << index_tpc_origin[2] << ")"
              << std::endl;
    std::cout << "index-tpc-end: ("
              << index_tpc_end[0] << ","
              << index_tpc_end[1] << ","
              << index_tpc_end[2] << ")"
              << std::endl;
    std::cout << "=====================================" << std::endl;

}



/**
 * @brief Create PMT positions tensor for all 32 PMTs
 * @return Torch tensor of shape (32, 3) with PMT positions
 */
torch::Tensor ModelInputInterface::_createPMTPositionsTensor() 
{
    torch::Tensor pmtpos = torch::zeros({32, 3}, torch::kFloat32);

    for (int i = 0; i < 32; ++i) {
        std::array<float,3> pos = flashmatch::PMTPositions::getOpDetPos(i);
        if ( pos[0]==0 && pos[1]==0 && pos[2]==0 ) {
            // returned a null position
            throw std::runtime_error("Got a bad PMT position!");
        }
        for (int j = 0; j < 3; ++j) {
            pmtpos[i][j] = pos[j];
        }
    }
    // // Adjust coordinate system to match 'tensor' system
    // // where y=0 is at bottom of TPC (shift by +117)
    // pmtpos.index({torch::indexing::Slice(), 1}) += 117.0;

    return pmtpos;
}

void ModelInputInterface::_define_pmt_positions()
{
    // Use the PMT positions utility to get actual detector positions
    _pmtpos = _createPMTPositionsTensor();  // use_v4_geom = true

    // The createPMTPositionsTensor function already handles the coordinate
    // system adjustment to match the 'tensor' system where y=0 is at bottom of TPC

    // Debug output
    std::cout << "PMT Positions initialized. Shape: ["
              << _pmtpos.size(0) << ", " << _pmtpos.size(1) << "]" << std::endl;
}

void ModelInputInterface::prepare_input_tensor(
        const std::vector< std::vector<float> >& hitpos_v,
        const std::vector< std::vector<float> >& hitimgcoord_v,
        const std::vector< larcv::Image2D >& wireplane_v,
        torch::Tensor& voxel_features,
        torch::Tensor& voxel_charge )
{
    // First determine which voxels to use
    flashmatch::dataprep::CosmicTrack dummy;
    dummy.hitpos_v = hitpos_v;
    dummy.hitimgpos_v = hitimgcoord_v;

    // Get voxel information
    std::vector< std::vector<int> > voxel_indices_vv;
    std::vector< std::vector<float> > voxel_centers_vv;
    std::vector< std::vector<float> > voxel_avepos_vv;
    std::vector< std::vector<float> > voxel_planecharge_vv;

    _assign_voxel_charge.makeVoxelChargeTensor(
        dummy,
        wireplane_v,
        _voxelizer,
        voxel_indices_vv,
        voxel_centers_vv,
        voxel_avepos_vv,
        voxel_planecharge_vv
    );

    // Convert to tensors
    int num_voxels = voxel_avepos_vv.size();
    if (num_voxels == 0) {
        // Return empty tensors if no voxels
        voxel_features = torch::zeros({0, 32, 8}, torch::kFloat32);
        voxel_charge = torch::zeros({0, 32, 1}, torch::kFloat32);
        return;
    }

    // Create coordinate tensor from average positions (N, 3)
    torch::Tensor coord = torch::zeros({num_voxels, 3}, torch::kFloat32);
    torch::Tensor planecharge = torch::zeros({num_voxels, 3}, torch::kFloat32);

    for (int i = 0; i < num_voxels; ++i) {
        for (int j = 0; j < 3; ++j) {
            coord[i][j] = voxel_avepos_vv[i][j];
            planecharge[i][j] = voxel_planecharge_vv[i][j];
        }
    }

    // Apply normalization to plane charges
    torch::Tensor planecharge_normalized = _normalize_planecharge(planecharge);

    // Prepare MLP input variables (reproducing prepare_mlp_input_variables)
    _prepare_mlp_input_variables(
        coord,
        planecharge_normalized,
        voxel_features,
        voxel_charge
    );
}

void ModelInputInterface::prepare_input_tensor(
        const larflow::voxelizer::VoxelChargeCalculator::VoxelChargeInfo_t& voxelchargeinfo,
        torch::Tensor& voxel_features,
        torch::Tensor& voxel_charge,
        torch::Tensor& mask )
{
    // Get voxel information in voxelchargeinfo
    // struct VoxelChargeInfo_t {
    //     float t0_assumed;
    //     int num_outside_tpc;
    //     std::vector< std::vector<int> >   voxel_indices_vv;
    //     std::vector< std::vector<float> > voxel_centers_vv;
    //     std::vector< std::vector<float> > voxel_avepos_vv;
    //     std::vector< std::vector<float> >  voxel_planecharge_vv;
    //     VoxelChargeInfo_t()
    //     : t0_assumed(0.0),
    //     num_outside_tpc(0) 
    //     {};
    // };

    // Convert to tensors
    int num_voxels = voxelchargeinfo.voxel_avepos_vv.size();
    if (num_voxels == 0) {
        // Return empty tensors if no voxels
        voxel_features = torch::zeros({0, 32, 8}, torch::kFloat32);
        voxel_charge = torch::zeros({0, 32, 1}, torch::kFloat32);
        return;
    }

    // Create coordinate tensor from average positions (N, 3)
    torch::Tensor coord = torch::zeros({num_voxels, 3}, torch::kFloat32);
    torch::Tensor planecharge = torch::zeros({num_voxels, 3}, torch::kFloat32);
    mask  = torch::zeros( {num_voxels,1}, torch::kFloat32 );

    for (int i = 0; i < num_voxels; ++i) {
        for (int j = 0; j < 3; ++j) {
            coord[i][j]       = voxelchargeinfo.voxel_avepos_vv[i][j];
            planecharge[i][j] = voxelchargeinfo.voxel_planecharge_vv[i][j];
            mask[i][0]         = 1.0;
        }
    }

    // Apply normalization to plane charges
    torch::Tensor planecharge_normalized = _normalize_planecharge(planecharge);

    // Prepare MLP input variables (reproducing prepare_mlp_input_variables)
    _prepare_mlp_input_variables(
        coord,
        planecharge_normalized,
        voxel_features,
        voxel_charge
    );
}

torch::Tensor ModelInputInterface::_normalize_planecharge(const torch::Tensor& planecharge)
{
    // Apply linear normalization: (charge + offset) / scale
    torch::Tensor normalized = torch::zeros_like(planecharge);

    for (int plane = 0; plane < 3; ++plane) {
        normalized.index({torch::indexing::Slice(), plane}) =
            (planecharge.index({torch::indexing::Slice(), plane}) + _planecharge_offset[plane]) / _planecharge_scale[plane];
    }

    return normalized;
}

void ModelInputInterface::_calc_dist_to_pmts(
    const torch::Tensor& pos_cm,
    torch::Tensor& dist2pmts,
    torch::Tensor& dvec2pmts)
{
    int n_voxels = pos_cm.size(0);
    int n_pmts = 32;

    // Initialize output tensors
    dist2pmts = torch::zeros({n_voxels, n_pmts, 1}, torch::kFloat32);
    dvec2pmts = torch::zeros({n_voxels, n_pmts, 3}, torch::kFloat32);

    // Calculate distances and vectors for each voxel-PMT pair
    for (int i = 0; i < n_voxels; ++i) {
        for (int j = 0; j < n_pmts; ++j) {
            // Calculate relative position vector
            float dx = pos_cm[i][0].item<float>() - _pmtpos[j][0].item<float>();
            float dy = pos_cm[i][1].item<float>() - _pmtpos[j][1].item<float>();
            float dz = pos_cm[i][2].item<float>() - _pmtpos[j][2].item<float>();

            // Store relative vector
            dvec2pmts[i][j][0] = dx;
            dvec2pmts[i][j][1] = dy;
            dvec2pmts[i][j][2] = dz;

            // Calculate and store distance
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            dist2pmts[i][j][0] = dist;
        }
    }
}

void ModelInputInterface::_prepare_mlp_input_variables(
    const torch::Tensor& coord,
    const torch::Tensor& q_perplane,
    torch::Tensor& voxel_features,
    torch::Tensor& voxel_charge)
{
    /**
     * Reproduces prepare_mlp_input_variables from Python
     * Creates for each (voxel, pmt) pair a 7-d input vector:
     * (x, y, z, dx, dy, dz, dist)
     * Plus the charge as a separate tensor
     */

    int nvoxels = coord.size(0);
    int npmt = 32;

    // Copy positions
    torch::Tensor detpos = coord;

    // Calculate distances and relative vectors to PMTs
    torch::Tensor dist2pmts, dvec2pmts;
    _calc_dist_to_pmts(detpos, dist2pmts, dvec2pmts);

    // Normalize positions and distances
    torch::Tensor detlens = torch::tensor({300.0, 300.0, 1500.0}, torch::kFloat32);
    detpos = detpos / detlens;  // Normalize detector positions

    // Scale distances (matching Python: dist / (210.0 * 5.0))
    dist2pmts = dist2pmts / (210.0 * 5.0);

    // Scale relative vectors
    dvec2pmts = dvec2pmts / detlens.unsqueeze(0).unsqueeze(0);  // Broadcast division

    // Create copies of coordinates for each PMT
    torch::Tensor detpos_perpmt = detpos.unsqueeze(1).repeat({1, npmt, 1});  // (N, 32, 3)

    // Concatenate features: [x, y, z, dx, dy, dz, dist]
    voxel_features = torch::cat({detpos_perpmt, dvec2pmts, dist2pmts}, 2);  // (N, 32, 7)

    // Prepare charge tensor: mean charge across planes, repeated for each PMT
    torch::Tensor q_mean = q_perplane.mean(1, true);  // Mean over planes: (N, 1)
    voxel_charge = q_mean.unsqueeze(1).repeat({1, npmt, 1});  // (N, 32, 1)

    // Combine features and charge for final input
    // The SIREN model expects them concatenated
    torch::Tensor combined = torch::cat({voxel_features, voxel_charge}, 2);  // (N, 32, 8)

    // Reshape for model input
    // The model expects (N*32, 7) for features and (N*32, 1) for charge
    int total_pairs = nvoxels * npmt;
    voxel_features = combined.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 7)});
    voxel_features = voxel_features.reshape({total_pairs, 7});

    voxel_charge = combined.index({torch::indexing::Slice(), torch::indexing::Slice(), 7});
    voxel_charge = voxel_charge.reshape({total_pairs, 1});
}

}