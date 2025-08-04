#ifndef __FLASHMATCH_DATAPREP_COSMICRECOINPUT_H__
#define __FLASHMATCH_DATAPREP_COSMICRECOINPUT_H__

/**
 * @brief Interface to the output ROOT File and Tree made by the Cosmic Reco
 * 
 * This class loads the ROOT file and FlashMatchData TTree.
 * 
 * Theocosmic reco code that defines and fills this tree is located in the ubdl/larflow repository at
 * `larflow/larflow/Reco/CosmicReco.h/cxx`.
 * 
 * We intend this class to help us load the input data to the flash match algorithm.
 * 
 */

#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"

#include "DataStructures.h"

#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/opflash.h"
#include "larlite/DataFormat/crttrack.h"
#include "larlite/DataFormat/crthit.h"
#include "larlite/DataFormat/larflowcluster.h"


namespace flashmatch {
namespace dataprep {

class CosmicRecoInput {

public:

    CosmicRecoInput( std::string inputfile );
    ~CosmicRecoInput();

    void load_entry( int ientry );
    void next_entry();
    
    // Getter methods for accessing event data
    int get_run() const { return run; }
    int get_subrun() const { return subrun; }
    int get_event() const { return event; }
    long get_num_entries() const { return _num_entries; }
    long get_current_entry() const { return _current_entry; }
    
    const std::vector<larlite::track>& get_track_v() const { return *_br_track_v; }
    const std::vector<larlite::opflash>& get_opflash_v() const { return *_br_opflash_v; }
    const std::vector<larlite::crttrack>& get_crttrack_v() const { return *_br_crttrack_v; }
    const std::vector<larlite::crthit>& get_crthit_v() const { return *_br_crthit_v; }

protected:

    std::string _input_file;
    TFile* _root_file;
    TTree* _flashmatchtree;
    long   _num_entries;
    long   _current_entry;

    // branches
    int run;
    int subrun;
    int event;

    std::vector<larlite::track>*    _br_track_v;
    std::vector<larlite::opflash>*  _br_opflash_v;
    std::vector<larlite::crttrack>* _br_crttrack_v;
    std::vector<larlite::crthit>*   _br_crthit_v;
    std::vector< std::vector< std::vector<float> > >* _br_trackhits_v;

    void _load_tfile_and_ttree();
    void _close();

};

}
}


#endif