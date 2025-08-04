#ifndef __FLASHMATCH_DATAPREP_LARLITE_DATA_INTERFACE_H__
#define __FLASHMATCH_DATAPREP_LARLITE_DATA_INTERFACE_H__

/**
 * @brief Functions for translating larlite data products into our data structures
 */

#include <string>
#include <vector>

#include "larlite/DataFormat/opflash.h"
#include "larlite/DataFormat/storage_manager.h"
#include "DataStructures.h"

namespace flashmatch {
namespace dataprep {

OpticalFlash convert_opflash( const larlite::opflash& opflash );

std::vector<OpticalFlash> convert_event_opflashes( 
    larlite::storage_manager& ioll,
    std::vector< std::string > opflash_src_treenames );

std::vector<OpticalFlash> convert_event_opflashes( const std::vector<larlite::opflash>& opflash_v );

}
}

#endif