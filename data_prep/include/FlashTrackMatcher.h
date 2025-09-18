#ifndef FLASHTRACKMATCHER_H
#define FLASHTRACKMATCHER_H

#include "DataStructures.h"
#include <string>
#include <map>

#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"

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
    ~FlashTrackMatcher();

    /**
     * @brief Find flash-track matches using Anode/Cathode crossing times
     * @param input_event_data  Event data containing tracks and flashes
     * @param output_match_data Event data containing matched tracks and flashes
     * @return number of matches
     */
    int FindAnodeCathodeMatches(const EventData& input_event_data,
                                EventData& output_match_data );

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
     * @brief Print matching statistics
     */
    void PrintStatistics();

private:
    FlashMatchConfig config_;               ///< Configuration parameters
    
    // Detector-specific parameters
    static constexpr double ANODE_X = 256.4;           ///< Anode X position [cm]
    static constexpr double CATHODE_X = 0.0;           ///< Cathode X position [cm]
    static constexpr int NUM_PMTS = 32;                ///< Number of PMTs
    static constexpr double DRIFT_VELOCITY = 0.109;    ///< Average drift velocity in UB (cm per usec)

    // Space Charge Utility: For correcting the space charge effect
    larutil::SpaceChargeMicroBooNE* _sce;

};

} // namespace dataprep
} // namespace flashmatch

#endif // FLASHTRACKMATCHER_H