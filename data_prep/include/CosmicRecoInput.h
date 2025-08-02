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

#include "TFile.h"
#include "TTree.h"

#include "DataStructures.h"


namespace flashmatch {
namespace dataprep {

class CosmicRecoInput {

public:

    CosmicRecoInput( std::string inputfile );
    ~CosmicRecoInput();

    void load_entry( int ientry );
    void next_entry();
    void get_entry_data( )

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
    std::vector<larlite::crthit>*   _bt_crthit_v;

    void _load_tfile_and_ttree();
    void _close();

};

}
}


#endif