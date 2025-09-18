#ifndef __FLASHMATCH_DATAPREP_PMT_POSITIONS__
#define __FLASHMATCH_DATAPREP_PMT_POSITIONS__

/**
 * @brief PMT position utilities for MicroBooNE detector
 */

#include <vector>
#include <array>

namespace flashmatch {

class PMTPositions {

protected:

    PMTPositions(){};
    ~PMTPositions(){};

    static std::vector< std::array<float,3> > _uboone_pmt_positions;
    static bool _loaded_array;
    static void load_pmtpos_array();

public:

    static int getNumPMTs() { return 32; };
    static int getNumPaddles() { return 4; };
    static int getNumPMTsAndPaddles() { return 36; };
    static std::array<float,3> getOpDetPos(int opdetid);

};



} // namespace flashmatch

#endif