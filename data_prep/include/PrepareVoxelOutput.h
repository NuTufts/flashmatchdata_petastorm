#ifndef __FLASHMATCHDATA_DATA_PREP_PREPARE_VOXEL_OUTPUT_H__
#define __FLASHMATCHDATA_DATA_PREP_PREPARE_VOXEL_OUTPUT_H__

#include <vector>
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larflow/Voxelizer/VoxelizeTriplets.h"
#include "DataStructures.h"

namespace flashmatch {
namespace dataprep {

class PrepareVoxelOutput {

public:

    PrepareVoxelOutput();
    ~PrepareVoxelOutput();

    /**
    * @brief Convert Cosmic Track Info into the charge voxel info used for network
    * 
    */
    int makeVoxelChargeTensor( 
        const CosmicTrack& cosmic_track, 
        const std::vector<larcv::Image2D>& adc_v,
        const larflow::voxelizer::VoxelizeTriplets& voxelizer, 
        std::vector< std::vector<int> >& voxel_indices_vv,
        std::vector< std::vector<float> >& voxel_centers_vv,
        std::vector< std::vector<float> >& voxel_avepos_vv,
        std::vector< std::vector<float> >& voxel_planecharge_vv );

    // Space Charge Utility: For correcting the space charge effect
    larutil::SpaceChargeMicroBooNE* _sce;    

};

}
}

#endif