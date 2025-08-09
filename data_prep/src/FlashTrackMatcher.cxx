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

int FlashTrackMatcher::FindAnodeCathodeMatches(const EventData& input_event_data, 
        EventData& output_match_data ) 
{
    auto const& optical_flashes = input_event_data.optical_flashes;
    auto const& cosmic_tracks   = input_event_data.cosmic_tracks;

    int num_matches = 0;

    struct MatchCandidate_t {

        double dt;
        int plane; // 0: anode, 1: cathode
        int itrack;
        int iflash;

        MatchCandidate_t()
        : dt(1e9), plane(-1), itrack(-1), iflash(-1)
        {};

        MatchCandidate_t( double dt_, int plane_, int itrack_, int iflash_ )
        : dt(dt_), plane(plane_), itrack(itrack_), iflash(iflash_)
        {};

        bool operator< ( const MatchCandidate_t& rhs ) const {
            if ( dt < rhs.dt ) {
                return true;
            }
            return false;
        };
    };

    for ( size_t itrack=0; itrack<cosmic_tracks.size(); itrack++ ) {
        
        auto const& cosmic_track = cosmic_tracks.at(itrack);

        int best_match = -1;
        float min_cathode_dt = 1e9;

        // we need bounds for track
        TVector3 pt_bounds[3][2];

        double bounds[3][2] = { {1e9,-1e9},{1e9,-1e9},{1e9,-1e9} };

        std::vector< MatchCandidate_t > candidates;

        for (auto const& segpt : cosmic_track.points ) {

            for (int idim=0; idim<3; idim++) {

                if ( segpt[idim]<bounds[idim][0] ) {
                    bounds[idim][0] = segpt[idim];
                    pt_bounds[idim][0] = segpt;
                }
                if ( segpt[idim]>bounds[idim][1] ) {
                    bounds[idim][1] = segpt[idim];
                    pt_bounds[idim][1] = segpt;
                }

            }
        }

        // now we look for anode/cathode crossing
        for ( size_t iflash=0; iflash<optical_flashes.size(); iflash++) {

            auto const& flash = optical_flashes.at(iflash);

            // Anode check
            double xmin = bounds[0][0];

            double xmin_anodetime = xmin/config_.drift_velocity;

            double dt_anode = std::fabs( xmin_anodetime-flash.flash_time );

            if ( dt_anode<2 ) {
                candidates.push_back( MatchCandidate_t(dt_anode, 0, itrack, iflash) );
            }


            // Cathod match: x-bounds
            double flash_time_cathode = flash.flash_time + 256.0/config_.drift_velocity;
            double xmax_time          = bounds[0][1]/config_.drift_velocity;
            double dt_cathode = std::fabs( xmax_time-flash_time_cathode);
            //std::cout << "track[" << itrack << "]-iflash[" << iflash << "] dt_cathode=" << dt_cathode << std::endl;
            if ( dt_cathode<5.0 ) {
                candidates.push_back( MatchCandidate_t(dt_cathode,1,itrack,iflash));
            }

        }

        if ( candidates.size()==0 )
            continue;

        std::sort( candidates.begin(), candidates.end() );
        
        // Loop over candidates, check overlap with (Z,Y)

        for ( auto const& cand : candidates ) {

            // check spatial match
            auto const& cand_flash = input_event_data.optical_flashes.at( cand.iflash );
            double flash_zmin = cand_flash.flash_center[2] - cand_flash.flash_width_z;
            double flash_zmax = cand_flash.flash_center[2] + cand_flash.flash_width_z;

            bool zoverlap = false;
            if ( bounds[2][0] >= flash_zmin && bounds[2][0] <= flash_zmax ) {
                zoverlap = true;
            }
            if ( bounds[2][1] >= flash_zmin && bounds[2][1] <= flash_zmax ) {
                zoverlap = true;
            }

            if ( cand.plane==0 ) {
                std::cout << "Anode Match ===============" << std::endl;
                std::cout << "  Track[" << cand.itrack << "]" << std::endl;
                std::cout << "  OpFlash[" << cand.iflash << "]" << std::endl;
                std::cout << "  dt: " << cand.dt << " usec" << std::endl;
            }
            else if ( cand.plane==1 ) {
                std::cout << "Cathode Match ===============" << std::endl;
                std::cout << "  Track[" << cand.itrack << "]" << std::endl;
                std::cout << "  OpFlash[" << cand.iflash << "]" << std::endl;
                std::cout << "  dt: " << cand.dt << " usec" << std::endl;
            }

            if ( !zoverlap ) {
                std::cout << "  No Z-overlap" << std::endl;
                continue;
            }
            else {
                std::cout << "  ** Make Match **" << std::endl;
            }

            // passes spatial check. make the match
            output_match_data.optical_flashes.push_back( cand_flash );

            CosmicTrack out_cosmictrack = cosmic_track; // Make a copy
            double x_t0_offset = cand_flash.flash_time*config_.drift_velocity;
            // shift the x location of the hits, now that we have a t0
            for (auto& hit : out_cosmictrack.hitpos_v ) {
                hit[0] -= x_t0_offset;
                // TODO: apply the Space Charge Effect correction, moving charge to correction position
                // Want a user-friendly utility in larflow::recoutils to do this I think
            }
            for (auto& hit : out_cosmictrack.points ) {
                hit[0] -= x_t0_offset;
            }
            out_cosmictrack.start_point[0] -= x_t0_offset;
            out_cosmictrack.end_point[0]   -= x_t0_offset;
            // note that the original imgpos are saved -- so we can go back and get the image charge
            output_match_data.cosmic_tracks.push_back( out_cosmictrack );

            // make empty crttrack and crthit
            output_match_data.crt_hits.push_back( CRTHit() );
            output_match_data.crt_tracks.push_back( CRTTrack() );

            num_matches++;

            break;
        }
    }

    return num_matches;
}

int FlashTrackMatcher::FindMatches(const EventData& input_data,
                                    EventData& output_data )
{

    std::set<int> matched_tracks;
    std::set<int> matched_flashes;

    for (size_t imatch=0; imatch<output_data.cosmic_tracks.size(); imatch++ ) {
        matched_tracks.insert(  output_data.cosmic_tracks.at(imatch).index );
        matched_flashes.insert( output_data.optical_flashes.at(imatch).index );
    }

    int num_matches_made = 0;

    // we could boot strap matches by scoring based on light-model estimate
    // struct MatchCandidate_t {
    //     int iflash;
    // }

    // Find all possible track-flash matches
    for (size_t track_idx = 0; track_idx < input_data.cosmic_tracks.size(); ++track_idx) {

        auto& cosmic_track = input_data.cosmic_tracks[track_idx];

        if ( cosmic_track.track_length<10.0 )
            continue;

        // we need bounds for track
        TVector3 pt_bounds[3][2];
        double bounds[3][2] = { {1e9,-1e9},{1e9,-1e9},{1e9,-1e9} };

        for (auto const& segpt : cosmic_track.points ) {

            for (int idim=0; idim<3; idim++) {

                if ( segpt[idim]<bounds[idim][0] ) {
                    bounds[idim][0] = segpt[idim];
                    pt_bounds[idim][0] = segpt;
                }
                if ( segpt[idim]>bounds[idim][1] ) {
                    bounds[idim][1] = segpt[idim];
                    pt_bounds[idim][1] = segpt;
                }

            }
        }

        // check image bounds -- no potentiall cut-off tracks
        double xmin_time = bounds[0][0]/config_.drift_velocity;
        double xmax_time = bounds[0][1]/config_.drift_velocity;

        if ( std::fabs(xmax_time-2635) < 10.0 ) {
            std::cout << "Cosmic Track[" << cosmic_track.index << "] is at late image boundary" << std::endl;
            continue;
        }
        if ( std::fabs(xmin_time+400.0) < 10.0 ) {
            std::cout << "Cosmic Track[" << cosmic_track.index << "] is at early image boundary" << std::endl;
            continue;
        }

        auto it_trackindex = matched_tracks.find( cosmic_track.index );
        if ( it_trackindex!=matched_tracks.end() ) {
            std::cout << "Cosmic Track[" << cosmic_track.index << "] is already matched" << std::endl;
            continue;
        }


        std::cout << "Search Flash Matches for Cosmic Track[" << cosmic_track.index << "] ---------" << std::endl;

        // find possible matches
        // the flash must fit within flash boundary
        // the flash must not also have been matched already
        // the flash must overlap with the z range of the flash as well
        int candidate_match_index = -1;
        int num_candidates = 0;
        
        for (size_t iflash=0; iflash<input_data.optical_flashes.size(); iflash++) {

            auto const& flash = input_data.optical_flashes.at(iflash);

            auto it_flashindex = matched_flashes.find( flash.index );

            if ( it_flashindex!=matched_flashes.end() ) {
                continue; // flash already used
            }

            double min_flashtime = flash.flash_time;
            double max_flashtime = flash.flash_time + 256.0/config_.drift_velocity;

            double xmin_time = bounds[0][0]/config_.drift_velocity;
            double xmax_time = bounds[0][1]/config_.drift_velocity;

            // std::cout << "cosmic[" << cosmic_track.index << "]-opflash[" << flash.index << "]" << std::endl;
            // std::cout << "  flash bounds: " << min_flashtime << ", " << max_flashtime << std::endl;
            // std::cout << "  xmin time: " << xmin_time << std::endl;
            // std::cout << "  xmax time: " << xmax_time << std::endl;

            bool is_inside_flashtime = false;
            if ( xmin_time >= min_flashtime && xmin_time <= max_flashtime 
                    && xmax_time >= min_flashtime && xmax_time <= max_flashtime ) 
            {
                is_inside_flashtime = true;
            }

            // check z-overlap
            double flash_zmin = flash.flash_center[2] - flash.flash_width_z*2.0;
            double flash_zmax = flash.flash_center[2] + flash.flash_width_z*2.0;

            bool zoverlap = false;
            if ( bounds[2][0] >= flash_zmin && bounds[2][0] <= flash_zmax ) {
                zoverlap = true;
            }
            if ( bounds[2][1] >= flash_zmin && bounds[2][1] <= flash_zmax ) {
                zoverlap = true;
            }

            // if ( zoverlap )
            //     std::cout << "  has z-overlap" << std::endl;
            // else
            //     std::cout << "  no z-overlap" << std::endl;

            if ( !is_inside_flashtime )
                continue;

            if ( !zoverlap )
                continue;

            std::cout << "CosmicTrack[" << cosmic_track.index << "]-OpFlash[" << flash.index << "] possible match" << std::endl;

            // possible match
            candidate_match_index = iflash;
            num_candidates++;
        }

        if ( num_candidates==1 ) {
            // unambiguos match is possible

            auto const& cand_flash = input_data.optical_flashes.at( candidate_match_index );
            output_data.optical_flashes.push_back( cand_flash );

            std::cout << "CosmicTrack[" << cosmic_track.index << "]-OpFlash[" << cand_flash.index << "] unambigious match" << std::endl;

            CosmicTrack out_cosmictrack = cosmic_track; // Make a copy
            double x_t0_offset = cand_flash.flash_time*config_.drift_velocity;
            // shift the x location of the hits, now that we have a t0
            for (auto& hit : out_cosmictrack.hitpos_v ) {
                hit[0] -= x_t0_offset;
                // TODO: apply the Space Charge Effect correction, moving charge to correction position
                // Want a user-friendly utility in larflow::recoutils to do this I think
            }
            for (auto& hit : out_cosmictrack.points ) {
                hit[0] -= x_t0_offset;
            }
            out_cosmictrack.start_point[0] -= x_t0_offset;
            out_cosmictrack.end_point[0]   -= x_t0_offset;
            // note that the original imgpos are saved -- so we can go back and get the image charge
            output_data.cosmic_tracks.push_back( out_cosmictrack );

            // make empty crttrack and crthit
            output_data.crt_hits.push_back( CRTHit() );
            output_data.crt_tracks.push_back( CRTTrack() );  
            num_matches_made++;          
        }
    }

    
    // // Update statistics
    // matched_tracks_ += unique_matches.size();
    
    // std::set<int> matched_flash_ids;
    // for (auto& match : unique_matches) {
    //     matched_flash_ids.insert(match.flash_id);
    //     if (match.has_crt_match) {
    //         crt_matched_tracks_++;
    //     }
    // }
    // matched_flashes_ += matched_flash_ids.size();
    
    return num_matches_made;
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