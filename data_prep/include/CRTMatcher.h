#ifndef CRTMATCHER_H
#define CRTMATCHER_H

#include "DataStructures.h"
#include <string>
#include <map>

#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"

namespace flashmatch {
namespace dataprep {

/**
 * @brief Class for matching cosmic ray tracks with CRT information
 * 
 * This class implements algorithms to associate cosmic ray tracks
 * with CRT (Cosmic Ray Tagger) hits and tracks for improved timing
 * and spatial constraints.
 */
class CRTMatcher {
public:
    /**
     * @brief Constructor
     */
    CRTMatcher();
    
    /**
     * @brief Destructor
     */
    ~CRTMatcher();
    
    /**
     * @brief Keep only CRT track objects that are in-time with optical flashes
     */
    int FilterCRTTracksByFlashMatches( 
        const std::vector< CRTTrack >& input_crt_tracks, 
        const std::vector< OpticalFlash >&input_opflashes );

    /**
     * @brief Keep only CRT hit objects that are in-time with optical flashes
     */
    int FilterCRTHitsByFlashMatches( 
        const std::vector< CRTHit >& input_crt_tracks, 
        const std::vector< OpticalFlash >&input_opflashes );
    
    /**
     * @brief Match a cosmic ray track to CRT tracks
     * @param cosmic_track The cosmic ray track
     * @param crt_tracks   Vector of CRT tracks
     * @param input_data   Class holding all the input data
     * @param output_data  Class holding the matches found
     * @return Index of best matching CRT track (-1 if no match)
     */
    int MatchToCRTTrack(CRTTrack& crt_track,
                        std::vector<CosmicTrack>& cosmic_tracks, 
                        const EventData& input_data,
                        EventData& output_data );
    
    /**
     * @brief Match a cosmic ray track to CRT hits
     * @param crt_hit     Vector of CRT hits.
     * @param input_data  All the event data, containing the cosmic tracks and opflash info we need.
     * @param output_Data Output event container. We'll put matches found into this object.
     * @return index of matched cosmic track
     */
    int MatchToCRTHits( const CRTHit& crthit, 
        const EventData& input_data, 
        EventData& output_data );
    
    
    
    /**
     * @brief Print CRT matching statistics
     */
    void PrintStatistics();

    /**
     * @brief set verbosity level
     * 
     * @param level Verbosity level where
     *      0: Quiet
     *      1: Normal
     *      2: Info
     *      3: Debug
     */
    void set_verbosity_level( int level )  { _verbosity = level; };

private:

    int _verbosity;                          ///< verbosity level
    enum { kQuiet=0, kNormal, kInfo, kDebug };

    // CRT matching statistics
    mutable int total_cosmic_tracks_;       ///< Total cosmic tracks processed
    mutable int crt_track_matches_;         ///< Number of CRT track matches
    mutable int crt_hit_matches_;           ///< Number of CRT hit matches
    mutable int total_crt_tracks_;          ///< Total CRT tracks processed
    mutable int total_crt_hits_;            ///< Total CRT hits processed

    std::map<int,int> _crttrack_index_to_flash_index;
    std::map<int,int> _crthit_index_to_flash_index;
    
    /**
     * @brief Update CRT matching statistics
     * @param crt_track_matched Whether CRT track match was found
     * @param crt_hit_matched Whether CRT hit matches were found
     */
    void UpdateStatistics(bool crt_track_matched, bool crt_hit_matched);

    static constexpr double DRIFT_VELOCITY = 0.1098; ///< cm per usec

    // Space Charge Utility: For correcting the space charge effect
    larutil::SpaceChargeMicroBooNE* _sce;
};

} // namespace dataprep
} // namespace flashmatch

#endif // CRTMATCHER_H