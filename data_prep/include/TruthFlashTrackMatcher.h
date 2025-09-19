#ifndef TRUTHFLASHTRACKMATCHER_H
#define TRUTHFLASHTRACKMATCHER_H

#include "DataStructures.h"
#include <string>
#include <map>
#include <vector>

// Forward declarations
namespace larcv {
    class EventImage2D;
    class Image2D;
}

namespace ublarcvapp {
namespace mctools {
    class FlashMatcherV2;
}
}

namespace larutil {
    class SpaceChargeMicroBooNE;
}

namespace flashmatch {
namespace dataprep {

/**
 * @brief Class for matching cosmic ray tracks to optical flashes using MC truth information
 *
 * This class implements a truth-based matching algorithm that:
 * 1. Projects track points back down into instance images
 * 2. Collects track IDs from the instance pixels
 * 3. Converts track IDs to votes for different flashes
 * 4. Matches tracks to flashes based on the voting results
 */
class TruthFlashTrackMatcher {
public:
    /**
     * @brief Constructor
     */
    TruthFlashTrackMatcher();

    /**
     * @brief Destructor
     */
    ~TruthFlashTrackMatcher();

    /**
     * @brief Perform truth-based flash-track matching
     * @param input_data Event data containing tracks and flashes
     * @param output_data Event data to store matched tracks and flashes
     * @param instance_img_v Instance images containing MC track IDs
     * @param truth_fm FlashMatcherV2 object with truth flash matches
     * @return Number of matches found
     */
    int MatchTracksToFlashes(const EventData& input_data,
                            EventData& output_data,
                            const std::vector<larcv::Image2D>& instance_img_v,
                            const ublarcvapp::mctools::FlashMatcherV2& truth_fm);

    /**
     * @brief Set verbosity level
     * @param level Verbosity level (0=quiet, 1=normal, 2=info, 3=debug)
     */
    void SetVerbosity(int level) { _verbosity = level; }

    void SetExcludeAnode(bool exclude) { _exclude_anode=exclude; };

    /**
     * @brief Print matching statistics
     */
    void PrintStatistics();

    /**
     * @brief Reset statistics
     */
    void ResetStatistics();

private:

    /**
     * @brief Project track points to wire planes and collect instance IDs
     * @param track Cosmic track to project
     * @param instance_img_v Instance images for each wire plane
     * @return Map of instance ID to vote count
     */
    std::map<int, int> CollectInstanceVotes(const CosmicTrack& track,
                                           const std::vector<larcv::Image2D>& instance_img_v);

    /**
     * @brief Convert instance IDs to flash indices using truth matching
     * @param instance_votes Map of instance ID to vote count
     * @param truth_fm FlashMatcherV2 object with truth flash matches
     * @return Map of flash index to vote count
     */
    std::map<int, int> ConvertToFlashVotes(const std::map<int, int>& instance_votes,
                                          const ublarcvapp::mctools::FlashMatcherV2& truth_fm);

    /**
     * @brief Find best flash match based on voting
     * @param flash_votes Map of flash index to vote count
     * @param min_vote_threshold Minimum votes required for a match
     * @return Index of best matching flash (-1 if no match)
     */
    int FindBestFlashMatch(const std::map<int, int>& flash_votes,
                         int min_vote_threshold = 10);

    /**
     * @brief Get pixel value from instance image at wire/tick location
     * @param img Instance image
     * @param wire Wire coordinate
     * @param tick Time tick coordinate
     * @return Instance ID at that pixel (or -1 if out of bounds)
     */
    int GetInstanceID(const larcv::Image2D& img, float wire, float tick);

    // Member variables
    int _verbosity;
    bool _exclude_anode;

    // Statistics
    int _total_tracks_processed;
    int _tracks_with_matches;
    int _total_flashes_matched;

    // Detector parameters
    static constexpr float DRIFT_VELOCITY = 0.1098; // cm/usec
    static constexpr float WIRE_PITCH[3] = {0.3, 0.3, 0.3}; // cm
    static constexpr float TICK_SAMPLING = 0.5; // usec
    static constexpr int NUM_WIRES[3] = {2400, 2400, 3456};
    static constexpr int NUM_TICKS = 6400;
    static constexpr float X_OFFSET = 0.0; // cm
    static constexpr float TRIG_TIME = 3200.0; // ticks
    // Space Charge Utility: For correcting the space charge effect
    larutil::SpaceChargeMicroBooNE* _sce;
};

} // namespace dataprep
} // namespace flashmatch

#endif // TRUTHFLASHTRACKMATCHER_H