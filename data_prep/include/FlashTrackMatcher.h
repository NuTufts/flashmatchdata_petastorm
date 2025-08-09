#ifndef FLASHTRACKMATCHER_H
#define FLASHTRACKMATCHER_H

#include "DataStructures.h"
#include <string>
#include <memory>

namespace flashmatch {
namespace dataprep {

/**
 * @brief Class for matching optical flashes with cosmic ray tracks
 * 
 * This class implements various algorithms to associate optical flashes
 * with cosmic ray tracks using timing, spatial, and CRT information.
 */
class FlashTrackMatcher {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters for flash-track matching
     */
    FlashTrackMatcher(FlashMatchConfig& config);

    /**
     * @brief Destructor
     */
    ~FlashTrackMatcher() = default;

    /**
     * @brief Find all flash-track matches in an event
     * @param event_data Event data containing tracks and flashes
     * @return Vector of flash-track matches
     */
    int FindMatches(const EventData& input_data, EventData& output_data );

    /**
     * @brief Find all flash-track matches in an event
     * @param input_event_data  Event data containing tracks and flashes
     * @param output_match_data Event data containing matched tracks and flashes
     * @return number of matches
     */
    int FindAnodeCathodeMatches(const EventData& input_event_data, 
                                EventData& output_match_data );

    /**
     * @brief Match a single track to the best flash
     * @param track The cosmic ray track
     * @param flashes Vector of optical flashes in the event
     * @param crt_tracks Vector of CRT tracks (optional)
     * @param crt_hits Vector of CRT hits (optional)
     * @return Best flash-track match (invalid if no match found)
     */
    FlashTrackMatch MatchTrackToFlash(CosmicTrack& track,
                                     std::vector<OpticalFlash>& flashes,
                                     std::vector<CRTTrack>& crt_tracks,
                                     std::vector<CRTHit>& crt_hits);


    /**
     * @brief Calculate expected anode crossing time for a track
     * @param track The cosmic ray track
     * @return Expected anode crossing time [μs]
     */
    double CalculateAnodeCrossingTime(CosmicTrack& track);

    /**
     * @brief Calculate expected cathode crossing time for a track
     * @param track The cosmic ray track
     * @return Expected cathode crossing time [μs]
     */
    double CalculateCathodeCrossingTime(CosmicTrack& track);

    /**
     * @brief Calculate spatial distance between track and flash
     * @param track The cosmic ray track
     * @param flash The optical flash
     * @return Spatial distance [cm]
     */
    double CalculateSpatialDistance(CosmicTrack& track, 
                                   OpticalFlash& flash);

    /**
     * @brief Calculate match score for a track-flash pair
     * @param track The cosmic ray track
     * @param flash The optical flash
     * @param time_diff Time difference [μs]
     * @param spatial_dist Spatial distance [cm]
     * @return Match score (higher is better)
     */
    double CalculateMatchScore(CosmicTrack& track,
                              OpticalFlash& flash,
                              double time_diff,
                              double spatial_dist);

    /**
     * @brief Find CRT track match for a cosmic ray track
     * @param cosmic_track The cosmic ray track
     * @param crt_tracks Vector of CRT tracks
     * @return Index of best matching CRT track (-1 if no match)
     */
    int FindCRTTrackMatch(CosmicTrack& cosmic_track,
                         std::vector<CRTTrack>& crt_tracks);

    /**
     * @brief Find CRT hit matches for a cosmic ray track
     * @param cosmic_track The cosmic ray track
     * @param crt_hits Vector of CRT hits
     * @return Vector of indices of matching CRT hits
     */
    std::vector<int> FindCRTHitMatches(CosmicTrack& cosmic_track,
                                      std::vector<CRTHit>& crt_hits);

    /**
     * @brief Resolve degeneracies when multiple tracks match the same flash
     * @param matches Vector of candidate matches
     * @return Vector of unique matches with degeneracies resolved
     */
    std::vector<FlashTrackMatch> ResolveDegeneracies(
        std::vector<FlashTrackMatch>& matches);

    /**
     * @brief Update configuration parameters
     * @param config New configuration parameters
     */
    void UpdateConfig(FlashMatchConfig& config);

    /**
     * @brief Get current configuration
     * @return Current configuration parameters
     */
    FlashMatchConfig& GetConfig() { return config_; }

    /**
     * @brief Load configuration from YAML file
     * @param filename Path to YAML configuration file
     * @return true if successfully loaded
     */
    bool LoadConfigFromFile(std::string& filename);

    /**
     * @brief Get matching statistics
     * @return Map of statistics
     */
    std::map<std::string, double> GetMatchingStatistics();

    /**
     * @brief Reset matching statistics
     */
    void ResetStatistics();
    
    /**
     * @brief Print matching statistics
     */
    void PrintStatistics();

private:
    FlashMatchConfig config_;               ///< Configuration parameters
    
    // Matching statistics
    mutable int total_tracks_;              ///< Total number of tracks processed
    mutable int matched_tracks_;            ///< Number of successfully matched tracks
    mutable int total_flashes_;             ///< Total number of flashes processed
    mutable int matched_flashes_;           ///< Number of flashes with matches
    mutable int crt_matched_tracks_;        ///< Number of tracks with CRT matches

    /**
     * @brief Check if timing is compatible between track and flash
     * @param track The cosmic ray track
     * @param flash The optical flash
     * @return true if timing is compatible
     */
    bool IsTimingCompatible(CosmicTrack& track, 
                           OpticalFlash& flash);

    /**
     * @brief Check if spatial distance is acceptable
     * @param track The cosmic ray track
     * @param flash The optical flash
     * @return true if spatial distance is acceptable
     */
    bool IsSpatiallyCompatible(CosmicTrack& track,
                              OpticalFlash& flash);

    /**
     * @brief Calculate PMT coverage for a track
     * @param track The cosmic ray track
     * @param flash The optical flash
     * @return PMT coverage fraction [0-1]
     */
    double CalculatePMTCoverage(CosmicTrack& track,
                               OpticalFlash& flash);

    /**
     * @brief Calculate track projection onto PMT array
     * @param track The cosmic ray track
     * @return Vector of expected PMT responses
     */
    std::vector<float> ProjectTrackToPMTs(CosmicTrack& track);

    /**
     * @brief Calculate chi-square between predicted and observed PE
     * @param predicted Vector of predicted PE values
     * @param observed Vector of observed PE values
     * @return Chi-square value
     */
    double CalculateChiSquare(std::vector<float>& predicted,
                              std::vector<float>& observed);
    
    /**
     * @brief Update matching statistics
     * @param track_matched Whether track was matched
     * @param flash_matched Whether flash was matched
     * @param crt_matched Whether CRT match was found
     */
    void UpdateMatchingStatistics(bool track_matched, 
                                 bool flash_matched, 
                                 bool crt_matched);
    
    // Detector-specific parameters
    static constexpr double ANODE_X = 256.4;           ///< Anode X position [cm]
    static constexpr double CATHODE_X = 0.0;           ///< Cathode X position [cm]
    static constexpr int NUM_PMTS = 32;                ///< Number of PMTs
    static constexpr double PMT_RESPONSE_THRESHOLD = 0.5; ///< Minimum PMT response threshold
    static constexpr double DRIFT_VELOCITY = 0.109;    ///< average drift velocity in UB (cm per usec)

};

} // namespace dataprep
} // namespace flashmatch

#endif // FLASHTRACKMATCHER_H