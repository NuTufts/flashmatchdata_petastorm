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
    FlashTrackMatcher(const FlashMatchConfig& config);
    
    /**
     * @brief Destructor
     */
    ~FlashTrackMatcher() = default;
    
    /**
     * @brief Find all flash-track matches in an event
     * @param event_data Event data containing tracks and flashes
     * @return Vector of flash-track matches
     */
    std::vector<FlashTrackMatch> FindMatches(const EventData& event_data);
    
    /**
     * @brief Match a single track to the best flash
     * @param track The cosmic ray track
     * @param flashes Vector of optical flashes in the event
     * @param crt_tracks Vector of CRT tracks (optional)
     * @param crt_hits Vector of CRT hits (optional)
     * @return Best flash-track match (invalid if no match found)
     */
    FlashTrackMatch MatchTrackToFlash(const CosmicTrack& track,
                                     const std::vector<OpticalFlash>& flashes,
                                     const std::vector<CRTTrack>& crt_tracks = {},
                                     const std::vector<CRTHit>& crt_hits = {});
    
    /**
     * @brief Calculate expected anode crossing time for a track
     * @param track The cosmic ray track
     * @return Expected anode crossing time [μs]
     */
    double CalculateAnodeCrossingTime(const CosmicTrack& track) const;
    
    /**
     * @brief Calculate expected cathode crossing time for a track
     * @param track The cosmic ray track
     * @return Expected cathode crossing time [μs]
     */
    double CalculateCathodeCrossingTime(const CosmicTrack& track) const;
    
    /**
     * @brief Calculate spatial distance between track and flash
     * @param track The cosmic ray track
     * @param flash The optical flash
     * @return Spatial distance [cm]
     */
    double CalculateSpatialDistance(const CosmicTrack& track, 
                                   const OpticalFlash& flash) const;
    
    /**
     * @brief Calculate match score for a track-flash pair
     * @param track The cosmic ray track
     * @param flash The optical flash
     * @param time_diff Time difference [μs]
     * @param spatial_dist Spatial distance [cm]
     * @return Match score (higher is better)
     */
    double CalculateMatchScore(const CosmicTrack& track,
                              const OpticalFlash& flash,
                              double time_diff,
                              double spatial_dist) const;
    
    /**
     * @brief Find CRT track match for a cosmic ray track
     * @param cosmic_track The cosmic ray track
     * @param crt_tracks Vector of CRT tracks
     * @return Index of best matching CRT track (-1 if no match)
     */
    int FindCRTTrackMatch(const CosmicTrack& cosmic_track,
                         const std::vector<CRTTrack>& crt_tracks) const;
    
    /**
     * @brief Find CRT hit matches for a cosmic ray track
     * @param cosmic_track The cosmic ray track
     * @param crt_hits Vector of CRT hits
     * @return Vector of indices of matching CRT hits
     */
    std::vector<int> FindCRTHitMatches(const CosmicTrack& cosmic_track,
                                      const std::vector<CRTHit>& crt_hits) const;
    
    /**
     * @brief Resolve degeneracies when multiple tracks match the same flash
     * @param matches Vector of candidate matches
     * @return Vector of unique matches with degeneracies resolved
     */
    std::vector<FlashTrackMatch> ResolveDegeneracies(
        const std::vector<FlashTrackMatch>& matches) const;
    
    /**
     * @brief Update configuration parameters
     * @param config New configuration parameters
     */
    void UpdateConfig(const FlashMatchConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration parameters
     */
    const FlashMatchConfig& GetConfig() const { return config_; }
    
    /**
     * @brief Load configuration from YAML file
     * @param filename Path to YAML configuration file
     * @return true if successfully loaded
     */
    bool LoadConfigFromFile(const std::string& filename);
    
    /**
     * @brief Get matching statistics
     * @return Map of statistics
     */
    std::map<std::string, double> GetMatchingStatistics() const;
    
    /**
     * @brief Reset matching statistics
     */
    void ResetStatistics();
    
    /**
     * @brief Print matching statistics
     */
    void PrintStatistics() const;

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
    bool IsTimingCompatible(const CosmicTrack& track, 
                           const OpticalFlash& flash) const;
    
    /**
     * @brief Check if spatial distance is acceptable
     * @param track The cosmic ray track
     * @param flash The optical flash
     * @return true if spatial distance is acceptable
     */
    bool IsSpatiallyCompatible(const CosmicTrack& track,
                              const OpticalFlash& flash) const;
    
    /**
     * @brief Calculate PMT coverage for a track
     * @param track The cosmic ray track
     * @param flash The optical flash
     * @return PMT coverage fraction [0-1]
     */
    double CalculatePMTCoverage(const CosmicTrack& track,
                               const OpticalFlash& flash) const;
    
    /**
     * @brief Calculate track projection onto PMT array
     * @param track The cosmic ray track
     * @return Vector of expected PMT responses
     */
    std::vector<double> ProjectTrackToPMTs(const CosmicTrack& track) const;
    
    /**
     * @brief Calculate chi-square between predicted and observed PE
     * @param predicted Vector of predicted PE values
     * @param observed Vector of observed PE values
     * @return Chi-square value
     */
    double CalculateChiSquare(const std::vector<double>& predicted,
                             const std::vector<double>& observed) const;
    
    /**
     * @brief Update matching statistics
     * @param track_matched Whether track was matched
     * @param flash_matched Whether flash was matched
     * @param crt_matched Whether CRT match was found
     */
    void UpdateMatchingStatistics(bool track_matched, 
                                 bool flash_matched, 
                                 bool crt_matched) const;
    
    // Detector-specific parameters
    static constexpr double ANODE_X = 256.4;           ///< Anode X position [cm]
    static constexpr double CATHODE_X = 0.0;           ///< Cathode X position [cm]
    static constexpr int NUM_PMTS = 32;                ///< Number of PMTs
    static constexpr double PMT_RESPONSE_THRESHOLD = 0.5; ///< Minimum PMT response threshold
};

} // namespace dataprep
} // namespace flashmatch

#endif // FLASHTRACKMATCHER_H