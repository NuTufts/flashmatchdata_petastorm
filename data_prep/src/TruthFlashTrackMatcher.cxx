/**
 * @file TruthFlashTrackMatcher.cxx
 * @brief Implementation of truth-based flash-track matching algorithms
 */

#include "TruthFlashTrackMatcher.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <set>

#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "ublarcvapp/MCTools/FlashMatcherV2.h"
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"

namespace flashmatch {
namespace dataprep {

// Define static constexpr members
constexpr float TruthFlashTrackMatcher::DRIFT_VELOCITY;
constexpr float TruthFlashTrackMatcher::WIRE_PITCH[3];
constexpr float TruthFlashTrackMatcher::TICK_SAMPLING;
constexpr int TruthFlashTrackMatcher::NUM_WIRES[3];
constexpr int TruthFlashTrackMatcher::NUM_TICKS;
constexpr float TruthFlashTrackMatcher::X_OFFSET;
constexpr float TruthFlashTrackMatcher::TRIG_TIME;

TruthFlashTrackMatcher::TruthFlashTrackMatcher()
    : _verbosity(1), 
    _exclude_anode(false),
    _total_tracks_processed(0),
    _tracks_with_matches(0), 
    _total_flashes_matched(0) 
{
    // create utility class that lets us go back to the "true" energy deposit location
    // before the space charge effect distortion
    _sce = new larutil::SpaceChargeMicroBooNE( larutil::SpaceChargeMicroBooNE::kMCC9_Backward );
}

TruthFlashTrackMatcher::~TruthFlashTrackMatcher() {
}

int TruthFlashTrackMatcher::MatchTracksToFlashes(const EventData& input_data,
                                                EventData& output_data,
                                                const std::vector<larcv::Image2D>& instance_img_v,
                                                ublarcvapp::mctools::FlashMatcherV2& truth_fm) {

    int num_matches = 0;

    // Copy event metadata
    output_data.run = input_data.run;
    output_data.subrun = input_data.subrun;
    output_data.event = input_data.event;

    // Track which flashes have been matched
    std::set<int> matched_flash_indices;

    // Process each cosmic track
    for (size_t track_idx = 0; track_idx < input_data.cosmic_tracks.size(); ++track_idx) {
        const auto& track = input_data.cosmic_tracks[track_idx];

        _total_tracks_processed++;

        if (_verbosity >= 2) {
            std::cout << "Processing track " << track_idx
                      << " with " << track.points.size() << " points"
                      << " and " << track.hitimgpos_v.size() << " hits" 
                      << std::endl;
        }

        //  Step 0: Filter Tracks

        // Skip very short tracks
        if (track.track_length < 10.0) {
            if (_verbosity >= 3) {
                std::cout << "  Skipping short track" << std::endl;
            }
            continue;
        }

	    // Skip tracks with only a few hits
	    if (track.hitimgpos_v.size()<3) {
	      if ( _verbosity>=3 )
	        std::cout << "  Skipping track with less than 3 htis" << std::endl;
	      continue;
	    }

        // Remove tracks near the edge to prevent cut-offs
        std::cout << "--------------------------------------" << std::endl;
        int num_image_edge_hits = 0;
        for ( auto const& hitcoord : track.hitimgpos_v ) {
            float tick = hitcoord[0];
            if ( tick<(2400.0+10*6.0) ) {
                num_image_edge_hits++;
                //std::cout << " edge tick: " << tick << std::endl;
            }
            else if ( tick>(2400.0+(1008-10)*6.0)) {
                num_image_edge_hits++;
                //std::cout << " edge tick: " << tick << std::endl;
            }                
        }
        std::cout << "NUM IMAGE EDGE HITS: " << num_image_edge_hits << std::endl;
        if ( num_image_edge_hits>0 )
            continue;

        // Step 1: Collect instance votes by projecting track points into instance images
        std::map<int, int> instance_votes = CollectInstanceVotes(track, instance_img_v);

        if (_verbosity >= 3) {
            std::cout << "  Collected votes from " << instance_votes.size()
                      << " unique instances" << std::endl;
            for (auto  it_instances=instance_votes.begin(); it_instances!=instance_votes.end(); it_instances++ )
                std::cout << "  id[" << it_instances->first << "] votes=" << it_instances->second << std::endl;
        }

        // Step 2: Convert instance IDs to flash indices using truth matching
        // the flash_votes dictionary is flashindex -> nvotes
        std::map<int, int> flash_votes = ConvertToFlashVotes(instance_votes, truth_fm);

        if (_verbosity >= 3) {
            std::cout << "  Converted to votes for " << flash_votes.size()
                      << " flashes" << std::endl;
        }

        // Step 3: Find best flash match based on voting
        // it returns the index of the flash pool stored in ublarcvapp::mctools::FlashMatcherV2& truth_fm
        int best_flash_idx = FindBestFlashMatch(flash_votes, 1);
    
        if (best_flash_idx >= 0 && best_flash_idx < (int)truth_fm.recoflash_v.size()) {

            // we need to match this to the flash object in the input_data.optical_flashes
            // containers. we match by time
            float recoflash_time_us = truth_fm.recoflash_v.at(best_flash_idx).time_us;
            float dt_min = 1e9;
            int best_match_opticalflash = -1;
            for (size_t iflash=0; iflash<input_data.optical_flashes.size(); iflash++) {
                float dt = std::fabs( recoflash_time_us - input_data.optical_flashes.at(iflash).flash_time );
                if ( dt < dt_min ) {
                    dt_min = dt;
                    best_match_opticalflash = iflash;
                }
            }

            if ( dt_min > 2.0 )
                continue;

            // Check if this flash has already been matched
            if (matched_flash_indices.find(best_match_opticalflash) == matched_flash_indices.end()) {

                if (_verbosity >= 2) {
                    std::cout << " -------------------------------------------------------------" << std::endl;
                    std::cout << "  Matched track " << track_idx
                              << " to flash " << best_flash_idx
                              << " with " << flash_votes[best_flash_idx] << " votes" << std::endl;
                    std::cout << "  FlashMatcherV2 flash time (us since trigger): " << recoflash_time_us << std::endl;
                    std::cout << "  input_data.optical_flashes flash time (us since trigger): " 
                              << input_data.optical_flashes.at(best_match_opticalflash).flash_time << std::endl;
                    std::cout << "  match dt: " << dt_min << " usec" << std::endl;
                }

                // Add the match to output
                CosmicTrack track_mod(track); // we create a copy because we will modify the 3D hit locations
                track_mod.sce_hitpos_v.clear();
                track_mod.sce_points.clear();

                int num_near_anode_hits = 0;
                int num_out_of_tpc_x = 0;

                for (size_t ihit=0; ihit<track_mod.hitpos_v.size(); ihit++) {
                    std::vector<float>& hit = track_mod.hitpos_v.at(ihit);
                    // for each position, we remove the t0 offset, now that we have the flash time
                    // then we apply the space charge effect correction
                    float dx = recoflash_time_us*DRIFT_VELOCITY;
                    hit[0] -= dx;

                    if ( hit[0]<0 )
                        num_out_of_tpc_x++;
                    else if (hit[0]>256.0)
                        num_out_of_tpc_x++;

                    bool applied_sce = false;
                    std::vector<double> hit_sce = _sce->ApplySpaceChargeEffect( hit[0], hit[1], hit[2], applied_sce);

                    if ( hit_sce[0]<10.0 ) {
                        num_near_anode_hits++;
                    }

                    track_mod.sce_hitpos_v.push_back( std::vector<float>{(float)hit_sce[0],(float)hit_sce[1],(float)hit_sce[2]} );        
                }
                std::cout << "Anode hits: " << num_near_anode_hits << std::endl;
                std::cout << "Num out of TPC: " << num_out_of_tpc_x << std::endl;
                std::cout << "dt_min: " << dt_min << std::endl;

                // adjust points along the reco'd line segment
                for (size_t ihit=0; ihit<track_mod.points.size(); ihit++) {
                    TVector3& hit = track_mod.points.at(ihit);
                    // for each position, we remove the t0 offset, now that we have the flash time
                    // then we apply the space charge effect correction
                    float dx = recoflash_time_us*DRIFT_VELOCITY;
                    hit[0] -= dx;

                    bool applied_sce = false;
                    std::vector<double> hit_sce = _sce->ApplySpaceChargeEffect( hit[0], hit[1], hit[2], applied_sce);
                    TVector3 vhit_sce( hit_sce[0], hit_sce[1], hit_sce[2] );
                    track_mod.sce_points.push_back( vhit_sce );        
                }

                // ---------------------------------------------------------
                // Determine if we keep the track-flash-match
                // Tests:
                //   1. number near anode
                //   2. is the truth track a muon?
                //   3. Does it pierce the node

                if ( _exclude_anode && num_near_anode_hits>0 ) {
                    std::cout << "Exclude track with anode hits: " << num_near_anode_hits << std::endl;
                    continue; 
                }

                if ( num_out_of_tpc_x>5 ) {
                    std::cout << "Exclude track because of number of out-of-tpc-x hits: " << num_out_of_tpc_x << std::endl;
                    continue;
                }

                // make sure these are downward going tracks that cross a TPC boundary
                // we look for the highest and lowest-z points.
                // the top should be a boundary point.
                // the last point should not be near the anode to avoid anode-piecers
                float top_z = -99999.0;
                float bot_z =  99999.0;
                std::vector< float > topz_pt(3,0);
                std::vector< float > botz_pt(3,0);
                for (size_t ihit=0; ihit<track_mod.hitpos_v.size(); ihit++) {
                    std::vector<float>& hit = track_mod.hitpos_v.at(ihit);
                    if ( hit[2] > top_z) {
                        top_z = hit[2];
                        topz_pt = hit;
                    }
                    if ( hit[2] < bot_z ) {
                        bot_z = hit[2];
                        botz_pt = hit;
                    }
                }
                // check if boundary
                float dwall_top = 100000.0;
                int top_border_index = -1;
                if ( topz_pt[0]>0 && topz_pt[0]<dwall_top ) {
                    dwall_top = topz_pt[0];
                    top_border_index = 0;
                }
                if ( topz_pt[0]<256.0 && (256.0-topz_pt[0])<dwall_top ) {
                    dwall_top = 256.0-topz_pt[0];
                    top_border_index = 0;
                }
                if ( topz_pt[1]>-116.5 && (topz_pt[1]+116.5)<dwall_top ) {
                    dwall_top = topz_pt[1]+116.5;
                    top_border_index = 1;
                }
                if ( topz_pt[1]<116.5 && (116.5-topz_pt[1])<dwall_top ) {
                    dwall_top = 116.5-topz_pt[1];
                    top_border_index = 1;
                }
                if ( topz_pt[2]>0.0 && (topz_pt[2])<dwall_top ) {
                    dwall_top = topz_pt[2];
                    top_border_index = 2;
                }
                if ( topz_pt[2]<1036.0 && (1036.0-topz_pt[2])<dwall_top ) {
                    dwall_top = 1036.0-topz_pt[2];
                    top_border_index = 2;
                }
                

                float dwall_bot = 100000.0;
                int bot_border_index = -1;
                if ( botz_pt[0]>0 && botz_pt[0]<dwall_bot ) {
                    dwall_bot = botz_pt[0];
                    bot_border_index = 0;
                }
                if ( botz_pt[0]<256.0 && (256.0-botz_pt[0])<dwall_bot ) {
                    dwall_bot = 256.0-botz_pt[0];
                    bot_border_index = 0;
                }
                if ( botz_pt[1]>-116.5 && (botz_pt[1]+116.5)<dwall_bot ) {
                    dwall_bot = botz_pt[1]+116.5;
                    bot_border_index = 1;
                }
                if ( botz_pt[1]<116.5 && (116.5-botz_pt[1])<dwall_bot ) {
                    dwall_bot = 116.5-botz_pt[1];
                    bot_border_index = 1;
                }
                if ( botz_pt[2]>0.0 && (botz_pt[2])<dwall_bot ) {
                    dwall_bot = botz_pt[2];
                    bot_border_index = 2;
                }
                if ( botz_pt[2]<1036.0 && (1036.0-botz_pt[2])<dwall_bot ) {
                    dwall_bot = 1036.0-botz_pt[2];
                    bot_border_index = 2;
                }

                if ( top_border_index==-1 || dwall_top > 10.0 ) {
                    std::cout << "Exclude track because top point is not on a boarder. dwall=" << dwall_top 
                              << " boarder_index=" << top_border_index << std::endl;
                    continue;
                }
                else {
                    std::cout << "passes top dwall requirement: dwall=" << dwall_top << " boarder_index=" << top_border_index << std::endl;
                }

                if ( bot_border_index==-1 || dwall_bot > 20.0 ) {
                    std::cout << "Exclude track because bot point is not on a boarder. dwall=" << dwall_bot
                              << " boarder_index=" << bot_border_index << std::endl;
                    continue;
                }

                if ( (top_border_index==0 && dwall_top<5.0) || (bot_border_index==0 && dwall_bot<5.0) ) {
                    std::cout << "Exclude track because either top or bottom point is at anode boarder" << std::endl;
                    std::cout << "  top dwall: " << dwall_top << " boarder_index=" << top_border_index << std::endl;
                    std::cout << "  bot dwall: " << dwall_bot << " boarder_index=" << bot_border_index << std::endl;
                    continue;
                }

                // we want to the get the largest truth instance ID
                auto const& recoflash = truth_fm.recoflash_v.at(best_flash_idx);
                std::vector<int> trackid_list = recoflash.trackid_list();

                // check if the largest contribution comes from a muon
                int max_instance_votes = 0;
                int max_particle_id = -1;
                for ( auto& trackid : trackid_list ) {
                    auto it_trackid_votes = instance_votes.find( trackid );
                    if ( it_trackid_votes!=instance_votes.end() ) {
                        int instance_votes = it_trackid_votes->second;
                        if ( instance_votes > max_instance_votes || (instance_votes>0 && max_particle_id<0) ) {
                            max_instance_votes = instance_votes;
                            // get the particle type
                            ublarcvapp::mctools::MCPixelPGraph::Node_t* ptracknode = truth_fm.mcpg.findTrackID( trackid );
                            if ( ptracknode != nullptr ) {
                                max_particle_id = ptracknode->pid;
                            }
                        }
                    }
                }
                std::cout << "Max particle ID matched: " << max_particle_id << " num of pixels matched = " << max_instance_votes << std::endl;
                if ( std::abs(max_particle_id)!=13 ) {
                    std::cout << "Exclude flash-track-match because it is not a muon: PID=" << max_particle_id << std::endl;
                    continue;
                }

                // ---------------------------------------------------------

                output_data.cosmic_tracks.emplace_back( std::move(track_mod) );
                output_data.optical_flashes.push_back(input_data.optical_flashes[best_match_opticalflash]);

                // Add empty CRT objects to maintain alignment
                output_data.crt_hits.push_back(CRTHit());
                output_data.crt_tracks.push_back(CRTTrack());

                // Mark as truth-based match (type 5)
                output_data.match_type.push_back(5);

                matched_flash_indices.insert(best_match_opticalflash);
                _tracks_with_matches++;
                num_matches++;
            }//if not already matched
        }
    }

    _total_flashes_matched = matched_flash_indices.size();

    if (_verbosity >= 1) {
        std::cout << "TruthFlashTrackMatcher: Found " << num_matches
                  << " matches from " << input_data.cosmic_tracks.size()
                  << " tracks and " << input_data.optical_flashes.size()
                  << " flashes" << std::endl;
    }

    return num_matches;
}

std::map<int, int> TruthFlashTrackMatcher::CollectInstanceVotes(const CosmicTrack& track,
                                                                const std::vector<larcv::Image2D>& instance_img_v) {
    std::map<int, int> instance_votes;

    // Check we have the expected 3 planes
    if (instance_img_v.size() != 3) {
        std::cerr << "Warning: Expected 3 instance images, got "
                  << instance_img_v.size() << std::endl;
        return instance_votes;
    }

    // Process each 3D point in the track
    for (size_t ipoint=0; ipoint<track.points.size(); ipoint++ ) {

        //const std::vector<float>& pos = track.hitpos_v.at(ipoint);
        const std::vector<float>& imgpos = track.hitimgpos_v.at(ipoint);

        // Project to each wire plane: each hit already has the image position
        for (int plane = 0; plane < 3; ++plane) {

            float tick = imgpos[0];
            float wire = imgpos[1+plane];

            // Get instance ID at this location
            int instance_id = GetInstanceID(instance_img_v[plane], wire, tick);

            if (instance_id > 0) {  // Valid instance ID (0 is background)
                instance_votes[instance_id]++;
            }
        }
    }

    return instance_votes;
}

std::map<int, int> TruthFlashTrackMatcher::ConvertToFlashVotes(const std::map<int, int>& instance_votes,
                                                              const ublarcvapp::mctools::FlashMatcherV2& truth_fm) {
    std::map<int, int> flash_votes;

    // For now, implement a simplified mapping since FlashMatcherV2 interface is unclear
    // This is a placeholder that maps instance IDs to flash indices directly
    // TODO: Implement proper truth flash matching interface

    // For each instance vote
    for (auto it = instance_votes.begin(); it != instance_votes.end(); ++it) {
        int instance_id = it->first; // track ID
        int vote_count = it->second; // counts

        // find flashes that have this trackid matched to it
        for ( size_t iflash=0; iflash<truth_fm.recoflash_v.size(); iflash++ ) {
            auto const& recoflash = truth_fm.recoflash_v.at(iflash);
            std::vector<int> trackid_list = recoflash.trackid_list();
            int matches = 0;
            for ( auto const& trackid : trackid_list ) {
                if ( trackid==instance_id ) {
                    matches++;
                }
            }

            if ( matches>0 ) {
                // we've found a match. pass the hit count votes to this flash!
                auto it = flash_votes.find( iflash );
                if ( it==flash_votes.end() ) {
                    flash_votes[ iflash] = 0;
                }
                flash_votes[ iflash] += vote_count;
            }
        }

    }

    return flash_votes;
}

int TruthFlashTrackMatcher::FindBestFlashMatch(const std::map<int, int>& flash_votes,
                                              int min_vote_threshold) {
    if (flash_votes.empty()) {
        return -1;
    }

    int best_flash_idx = -1;
    int max_votes = 0;

    for (auto it = flash_votes.begin(); it != flash_votes.end(); ++it) {
        int flash_idx = it->first;
        int vote_count = it->second;
        if (vote_count > max_votes && vote_count >= min_vote_threshold) {
            max_votes = vote_count;
            best_flash_idx = flash_idx;
        }
    }

    std::cout << "  flash with most votes [" << best_flash_idx << "] num votes=" << max_votes << std::endl;

    return best_flash_idx;
}

int TruthFlashTrackMatcher::GetInstanceID(const larcv::Image2D& img, float wire, float tick) {
    // Check bounds
    if (wire < 0 || wire >= img.meta().max_x() ||
        tick < 0 || tick >= img.meta().max_y()) {
        return -1;
    }

    // Convert to integer indices
    int wire_idx = (int)img.meta().col(wire);
    int tick_idx = (int)img.meta().row(tick);

    // Get pixel value (instance ID)
    float pixel_value = img.pixel(tick_idx, wire_idx);

    // Round to nearest integer (instance IDs should be integers)
    int instance_id = (int)std::round(pixel_value);

    return instance_id;
}

void TruthFlashTrackMatcher::PrintStatistics() {
    std::cout << "TruthFlashTrackMatcher Statistics:" << std::endl;
    std::cout << "  Total tracks processed: " << _total_tracks_processed << std::endl;
    std::cout << "  Tracks with matches: " << _tracks_with_matches << std::endl;
    std::cout << "  Total flashes matched: " << _total_flashes_matched << std::endl;

    if (_total_tracks_processed > 0) {
        double match_efficiency = 100.0 * _tracks_with_matches / _total_tracks_processed;
        std::cout << "  Match efficiency: " << match_efficiency << "%" << std::endl;
    }
}

void TruthFlashTrackMatcher::ResetStatistics() {
    _total_tracks_processed = 0;
    _tracks_with_matches = 0;
    _total_flashes_matched = 0;
}

} // namespace dataprep
} // namespace flashmatch
