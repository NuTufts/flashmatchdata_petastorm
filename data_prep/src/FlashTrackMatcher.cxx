/**
 * @file FlashTrackMatcher.cxx
 * @brief Implementation of flash-track matching algorithms
 */

#include "FlashTrackMatcher.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <set>

namespace flashmatch {
namespace dataprep {

// Define static constexpr members
constexpr double FlashTrackMatcher::ANODE_X;
constexpr double FlashTrackMatcher::CATHODE_X;
constexpr int FlashTrackMatcher::NUM_PMTS;
constexpr double FlashTrackMatcher::PMT_RESPONSE_THRESHOLD;

FlashTrackMatcher::FlashTrackMatcher(FlashMatchConfig& config)
    : config_(config), total_tracks_(0), matched_tracks_(0), 
      total_flashes_(0), matched_flashes_(0), crt_matched_tracks_(0) {
}

std::vector<FlashTrackMatch> FlashTrackMatcher::FindMatches(EventData& event_data) {
    std::vector<FlashTrackMatch> all_matches;

    total_tracks_ += event_data.cosmic_tracks.size();
    total_flashes_ += event_data.optical_flashes.size();

    // Find all possible track-flash matches
    for (size_t track_idx = 0; track_idx < event_data.cosmic_tracks.size(); ++track_idx) {
        auto& track = event_data.cosmic_tracks[track_idx];

        FlashTrackMatch best_match = MatchTrackToFlash(track, event_data.optical_flashes, 
                                                      event_data.crt_tracks, event_data.crt_hits);

        if (best_match.flash_id >= 0) {
            best_match.track_id = track_idx;
            all_matches.push_back(best_match);
        }
    }

    // Resolve degeneracies (multiple tracks matching same flash)
    std::vector<FlashTrackMatch> unique_matches = ResolveDegeneracies(all_matches);
    
    // Update statistics
    matched_tracks_ += unique_matches.size();
    
    std::set<int> matched_flash_ids;
    for (auto& match : unique_matches) {
        matched_flash_ids.insert(match.flash_id);
        if (match.has_crt_match) {
            crt_matched_tracks_++;
        }
    }
    matched_flashes_ += matched_flash_ids.size();
    
    return unique_matches;
}

FlashTrackMatch FlashTrackMatcher::MatchTrackToFlash(CosmicTrack& track,
                                                    std::vector<OpticalFlash>& flashes,
                                                    std::vector<CRTTrack>& crt_tracks,
                                                    std::vector<CRTHit>& crt_hits) {
    
    FlashTrackMatch best_match;
    double best_score = -1.0;

    for (size_t flash_idx = 0; flash_idx < flashes.size(); ++flash_idx) {
        auto& flash = flashes[flash_idx];

        // Check basic compatibility
        if (!IsTimingCompatible(track, flash) || !IsSpatiallyCompatible(track, flash)) {
            continue;
        }

        // Calculate match metrics
        double time_diff_anode = std::abs(flash.flash_time - track.anode_crossing_time);
        double time_diff_cathode = std::abs(flash.flash_time - track.cathode_crossing_time); //why is this set up this way?
        double time_diff = std::min(time_diff_anode, time_diff_cathode); // This definitely should not be done like this

        double spatial_dist = CalculateSpatialDistance(track, flash);
        double match_score = CalculateMatchScore(track, flash, time_diff, spatial_dist);

        // Compare match; if better, switch
        if (match_score > best_score) {
            best_score = match_score;
            best_match.flash_id = flash_idx;
            best_match.time_difference = time_diff;
            best_match.spatial_distance = spatial_dist;
            best_match.match_score = match_score;

            // Calculate PE prediction residual (simplified)
            std::vector<float> predicted_pe = ProjectTrackToPMTs(track);
            best_match.pe_prediction_residual = CalculateChiSquare(predicted_pe, flash.pe_per_pmt);
        }
    }

    // Add CRT information if available and matching is enabled
    if (best_match.flash_id >= 0 && config_.enable_crt_track_matching && !crt_tracks.empty()) {
        // TODO: Implement CRT matching
        // For now, just mark as not having CRT match
        best_match.has_crt_match = false;
    }

    return best_match;
}

double FlashTrackMatcher::CalculateAnodeCrossingTime(CosmicTrack& track) {
    // Calculate time when track crosses anode (X = ANODE_X)
    // This is a simplified calculation - real implementation would use proper drift time
    
    double anode_distance = std::abs(track.start_point.X() - ANODE_X);
    if (track.direction.X() != 0) {
        return anode_distance / (config_.drift_velocity * std::abs(track.direction.X()));
    }
    
    return track.anode_crossing_time; // Use existing value if available
}

//bool FlashTrackMatcher::anodeOrCathodeCrosser(CosmicTrack& track) {
    // Determine whether or not the track is a cathode crosser

    // First, assume it crossed the cathode. Find what time we would expect the flash to occur
//    double cathode_distance = std::abs(track.start_point.X() - CATHODE_X);
//    double time_since_flash = (cathode_distance/ config_.drift_velocity) // If it really is a cathode crosser, there should be a flash this many seconds earlier

    // Now do the same for the anode
//    double anode_distance = std::abs(track.start_point.X() - ANODE_X);
//    double time_until_flash = (anode_distance/ config_.drift_velocity); // If it's an anode crosser, the flash will happen in the near future

//    for (size_t flash_idx = 0; flash_idx < flashes.size(); ++flash_idx) {
//        auto& flash = flashes[flash_idx];

        // Check basic compatibility (this code is frankly suspect)
//        if (!IsTimingCompatible(track, flash) || !IsSpatiallyCompatible(track, flash)) {
//            continue;
//        }
//        if (std::abs((currentTime - flash.flash_time) - (currentTime - time_since_flash)) <= + 5) { // See if the flash happened within 5 miliseconds of our expectation
//            best_match.flash_id = flash_idx;
//            return true;
//        }

        // If we're still running the function, it wasn't a cathode crosser. Try anode?
//        if (std::abs((flash.flash_time - currentTime) - (time_until_flash - currentTime)) <= + 5) { // See if the flash happened within 5 miliseconds of our expectation
//            best_match.flash_id = flash_idx;
//            return true;
//        }

//        else {
//            return false;
//        }

//}


double FlashTrackMatcher::CalculateCathodeCrossingTime(CosmicTrack& track) {
    // Calculate time when track crosses cathode (X = CATHODE_X)
    
    double cathode_distance = std::abs(track.start_point.X() - CATHODE_X);
    if (track.direction.X() != 0) {
        return cathode_distance / (DRIFT_VELOCITY * std::abs(track.direction.X()));
    }
    
    return track.cathode_crossing_time; // Use existing value if available
}

double FlashTrackMatcher::CalculateSpatialDistance(CosmicTrack& track, 
                                                  OpticalFlash& flash) {
    // Calculate distance between track center and flash center
    TVector3 track_center = (track.start_point + track.end_point) * 0.5;
    return (track_center - flash.flash_center).Mag();
}

double FlashTrackMatcher::CalculateMatchScore(CosmicTrack& track,
                                             OpticalFlash& flash,
                                             double time_diff,
                                             double spatial_dist) {
    // Simple scoring function - can be made more sophisticated
    double time_score = std::exp(-time_diff / config_.anode_crossing_tolerance);
    double spatial_score = std::exp(-spatial_dist / config_.track_flash_distance_cut);
    double pe_score = 1.0; // Could add PE-based scoring
    
    return time_score * spatial_score * pe_score;
}

int FlashTrackMatcher::FindCRTTrackMatch(CosmicTrack& cosmic_track,
                                        std::vector<CRTTrack>& crt_tracks) {
    // TODO: Implement CRT track matching
    // For now, return no match
    return -1;
}

std::vector<int> FlashTrackMatcher::FindCRTHitMatches(CosmicTrack& cosmic_track,
                                                     std::vector<CRTHit>& crt_hits) {
    // TODO: Implement CRT hit matching
    // For now, return empty vector
    return {};
}

std::vector<FlashTrackMatch> FlashTrackMatcher::ResolveDegeneracies(
    std::vector<FlashTrackMatch>& matches) {

    std::vector<FlashTrackMatch> unique_matches;
    std::map<int, std::vector<size_t>> flash_to_matches;

    // Group matches by flash ID
    for (size_t i = 0; i < matches.size(); ++i) {
        flash_to_matches[matches[i].flash_id].push_back(i);
    }

    // For each flash, keep only the best match
    for (auto& flash_group : flash_to_matches) {
        if (flash_group.second.size() == 1) {
            // No degeneracy
            unique_matches.push_back(matches[flash_group.second[0]]);
        } else {
            // Multiple tracks match this flash - keep the best one
            size_t best_match_idx = flash_group.second[0];
            double best_score = matches[best_match_idx].match_score;

            for (size_t i = 1; i < flash_group.second.size(); ++i) {
                size_t match_idx = flash_group.second[i];
                if (matches[match_idx].match_score > best_score) {
                    best_score = matches[match_idx].match_score;
                    best_match_idx = match_idx;
                }
            }

            unique_matches.push_back(matches[best_match_idx]);
        }
    }
    
    return unique_matches;
}

void FlashTrackMatcher::UpdateConfig(FlashMatchConfig& config) {
    config_ = config;
}

bool FlashTrackMatcher::LoadConfigFromFile(std::string& filename) {
    // TODO: Implement YAML configuration loading
    std::cout << "Loading flash matching configuration from: " << filename << std::endl;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open configuration file: " << filename << std::endl;
        return false;
    }
    
    // TODO: Parse YAML and update config_
    return true;
}

std::map<std::string, double> FlashTrackMatcher::GetMatchingStatistics() {
    std::map<std::string, double> stats;
    
    stats["total_tracks"] = total_tracks_;
    stats["matched_tracks"] = matched_tracks_;
    stats["total_flashes"] = total_flashes_;
    stats["matched_flashes"] = matched_flashes_;
    stats["crt_matched_tracks"] = crt_matched_tracks_;
    
    if (total_tracks_ > 0) {
        stats["track_matching_efficiency"] = static_cast<double>(matched_tracks_) / total_tracks_;
    }
    
    if (total_flashes_ > 0) {
        stats["flash_matching_efficiency"] = static_cast<double>(matched_flashes_) / total_flashes_;
    }
    
    return stats;
}

void FlashTrackMatcher::ResetStatistics() {
    total_tracks_ = 0;
    matched_tracks_ = 0;
    total_flashes_ = 0;
    matched_flashes_ = 0;
    crt_matched_tracks_ = 0;
}

void FlashTrackMatcher::PrintStatistics() {
    auto stats = GetMatchingStatistics();

    std::cout << "Flash Matching Statistics:" << std::endl;
    std::cout << "  Total tracks: " << static_cast<int>(stats["total_tracks"]) << std::endl;
    std::cout << "  Matched tracks: " << static_cast<int>(stats["matched_tracks"]) << std::endl;
    std::cout << "  Total flashes: " << static_cast<int>(stats["total_flashes"]) << std::endl;
    std::cout << "  Matched flashes: " << static_cast<int>(stats["matched_flashes"]) << std::endl;
    std::cout << "  CRT matched tracks: " << static_cast<int>(stats["crt_matched_tracks"]) << std::endl;

    if (stats.find("track_matching_efficiency") != stats.end()) {
        std::cout << "  Track matching efficiency: " << stats["track_matching_efficiency"] * 100.0 << "%" << std::endl;
    }

    if (stats.find("flash_matching_efficiency") != stats.end()) {
        std::cout << "  Flash matching efficiency: " << stats["flash_matching_efficiency"] * 100.0 << "%" << std::endl;
    }
}

bool FlashTrackMatcher::IsTimingCompatible(CosmicTrack& track, 
                                          OpticalFlash& flash) {
    double time_diff_anode = std::abs(flash.flash_time - track.anode_crossing_time);
    double time_diff_cathode = std::abs(flash.flash_time - track.cathode_crossing_time);
    double min_time_diff = std::min(time_diff_anode, time_diff_cathode);
    
    return min_time_diff <= std::max(config_.anode_crossing_tolerance, config_.cathode_crossing_tolerance);
}

bool FlashTrackMatcher::IsSpatiallyCompatible(CosmicTrack& track,
                                             OpticalFlash& flash) {
    double spatial_dist = CalculateSpatialDistance(track, flash);
    return spatial_dist <= config_.track_flash_distance_cut;
}

double FlashTrackMatcher::CalculatePMTCoverage(CosmicTrack& track,
                                              OpticalFlash& flash) {
    // TODO: Implement PMT coverage calculation
    // For now, return dummy value
    return 0.5;
}

std::vector<float> FlashTrackMatcher::ProjectTrackToPMTs(CosmicTrack& track) {
    // TODO: Implement track projection to PMTs
    // For now, return uniform distribution
    std::vector<float> predicted_pe(NUM_PMTS, track.total_charge / NUM_PMTS);
    return predicted_pe;
}

double FlashTrackMatcher::CalculateChiSquare(std::vector<float>& predicted,
                                            std::vector<float>& observed) {
    if (predicted.size() != observed.size()) return 999.0;
    
    double chi_square = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        double diff = predicted[i] - observed[i];
        double error = std::sqrt(observed[i] + 1.0); // Poisson error
        chi_square += (diff * diff) / (error * error);
    }

    return chi_square;
}

void FlashTrackMatcher::UpdateMatchingStatistics(bool track_matched, 
                                                 bool flash_matched, 
                                                 bool crt_matched) {
    // Statistics are updated in FindMatches method
}

} // namespace dataprep
} // namespace flashmatch