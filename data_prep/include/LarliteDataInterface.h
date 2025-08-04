#ifndef __FLASHMATCH_DATAPREP_LARLITE_DATA_INTERFACE_H__
#define __FLASHMATCH_DATAPREP_LARLITE_DATA_INTERFACE_H__

/**
 * @brief Functions for translating larlite data products into our data structures
 */

#include <string>
#include <vector>

#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/opflash.h"
#include "larlite/DataFormat/crttrack.h"
#include "larlite/DataFormat/crthit.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/storage_manager.h"
#include "DataStructures.h"

namespace flashmatch {
namespace dataprep {

OpticalFlash convert_opflash( const larlite::opflash& opflash );

std::vector<OpticalFlash> convert_event_opflashes( 
    larlite::storage_manager& ioll,
    std::vector< std::string > opflash_src_treenames );

std::vector<OpticalFlash> convert_event_opflashes( const std::vector<larlite::opflash>& opflash_v );

CosmicTrack convert_trackinfo( 
    const larlite::track& track,
    const std::vector< std::vector<float> >& hitinfo );

std::vector< CosmicTrack > convert_event_trackinfo( 
    const std::vector< larlite::track >& track_list,
    const std::vector< std::vector< std::vector<float> > >& hitinfo_list );

}
}

#endif