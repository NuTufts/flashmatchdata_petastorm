#include "ModelInputInterface.h"

#include "DataStructures.h"

namespace flashmatch {

ModelInputInterface::ModelInputInterface()
{
    _voxel_len_cm = 5.0;
    _define_voxels();
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

void ModelInputInterface::prepare_input_tensor( 
        const std::vector< std::vector<float> >& hitpos_v, 
        const std::vector< std::vector<float> >& hitimgcoord_v,
        const std::vector< larcv::Image2D >& wireplane_v,
        torch::Tensor& voxel_features,
        torch::Tensor& voxel_charge )
{

    // first thing we need to do is determine which voxels 
    // we want to make use of the PrepareVoxelOutput class,
    // so we copy values into a dummy CosmicTrack object
    flashmatch::dataprep::CosmicTrack dummy;
    dummy.hitpos_v = hitpos_v; // just copy for now
    dummy.hitimgpos_v = hitimgcoord_v;

    // we will get back a list of voxels
    std::vector< std::vector<int> > voxel_indices_vv;       // indices of occupied voxels
    std::vector< std::vector<float> > voxel_centers_vv;     // center of occupied voxels
    std::vector< std::vector<float> > voxel_avepos_vv;      // average pos of points inside each occupied voxels
    std::vector< std::vector<float> > voxel_planecharge_vv; // charge from each plane for each occupied voxels
    _assign_voxel_charge.makeVoxelChargeTensor( 
        dummy,
        wireplane_v,
        _voxelizer,
        voxel_indices_vv,
        voxel_centers_vv,
        voxel_avepos_vv,
        voxel_planecharge_vv
    );

    // now we we have to prepare the input tensor
    // TODO: reproduce the action of the function
    // def prepare_input(...) from run_siren_inference.py
    // this involves also reproducing the behavior of the function
    // def prepare_mlp_input_variables(...) found in flashmatchnet/utils/coord_and_embed_functions.py

    // we prepare tensors
    //   voxel_features
    //   voxel_charge

    return;
}

}