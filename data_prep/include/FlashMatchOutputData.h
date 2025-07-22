#ifndef __FLASHMATCH_DATAPREP_FLASHMATCHOUTPUTDATA_H__
#define __FLASHMATCH_DATAPREP_FLASHMATCHOUTPUTDATA_H__

#include <array>
#include <vector>
#include <string>

#include "TFile.h"
#include "TTree.h"

#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/opflash.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/crthit.h"
#include "larlite/DataFormat/crttrack.h"

namespace flashmatch {
namespace dataprep {

/**
 * @brief Stores the output data for each event and manages IO to output rootfile
 */

  class FlashMatchOutputData {

  public:

    FlashMatchOutputData( std::string output_rootfile, bool allow_overwrite );
    ~FlashMatchOutputData();

    typedef std::array< float, 4 > CRTHitPos_t;  ///< defining an alias for CRT Hit (x,y,z,t)
    typedef std::array< CRTHitPos_t, 2 > CRTMatchLine_t; ///< a line that represents path between 
                                                         ///< two crthits or crthit and trackendpoint: 
                                                         // line used to test if energy deposit hits consistent
                                                         // with crossing or stopping muon path.
    CRTHitPos_t    nullCRTHitPos();    ///< make an instance of CRTHitPos_t that represents a null hit
    CRTMatchLine_t nullCRTMatchLine(); ///< make an instance of CRTMatchLine_t that represents a null match line
    bool isNullCRTHitPos( CRTHitPos_t& hit );  ///< test if hitpos object is a null-hit
    bool isNullCRTMatchLine( CRTMatchLine_t& matchline ); ///< test if matchline is a null-line

    // Output of flash matching algorithms   
    // These are containers, holding matches for one event at a time.
    int run;
    int subrun;
    int event;
    std::vector< larlite::track >           track_v;      //< line-segment representation of cosmic muon track
    std::vector< larlite::larflowcluster >  track_hits_v; //< 3d charge deposits associated to each track
    std::vector< larlite::opflash >         opflash_v;    //< opflashes matched to each track
    std::vector< CRTMatchLine_t  >          crtmatch_v;   //< if found, indicates a crt match. if both line endpoints is (0,0,0), 
                                                          //  this indicates a null-match.

    bool setRSE( int arun, int asubrun, int anevent ) {
        if ( arun==run && asubrun==subrun && anevent==event ) {
            // the run, subrun, event has not changed
            return false;
        }
        run    = arun;
        subrun = asubrun;
        event  = anevent;
        return true;
    };

    /// add a track-flash match to the event container: this is what algorithms should call to store a match
    int addTrackFlashMatch( larlite::track& track,  
                            larlite::opflash& opflash, 
                            larlite::larflowcluster* hitcluster=nullptr,
                            CRTMatchLine_t* crtmatch=nullptr );

    void clear(); ///< clear event containers

    // Output ROOT file 
    TFile* _file; ///< TFile object for output file

    // Output ROOT tree: will save one track-flash match per entry, in format closer to training
    TTree* _matched_tree; ///< TTree containing the output of the flashmatcher
    int matchindex;                                      ///< index of the track_v and opflash_v match in the event
    std::vector< std::vector<float> > track_segments_v;  ///< 3d points along a set of line segments representing one track
    std::vector< std::vector<float> > track_hitpos_v;    ///< 3d charge deposit points associated with the track
    std::vector< std::vector<float> > track_hitfeat_v;   ///< features (e.g. plane charge) per hit pos for the track
    std::vector< float >              opflash_pe_v;      ///< PE per pmt (index follows opdet ID)
    std::vector< std::vector<float> > crtmatch_endpts_v; ///< End points if we made a CRT match. Is [ (0,0,0,0), (0,0,0,0) ] if no CRT match.

    void makeMatchTTree();
    int  saveEventMatches(); ///< saves items in event container to the TTree using the output branch variables
    void writeTree(); ///< save ttree to the tfile
    void closeFile(); ///< destory tree object and close the tfile


  };


}
}

#endif