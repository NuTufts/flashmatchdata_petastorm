/**
 * @file CRTMatcher.cxx
 * @brief Implementation of CRT matching algorithms
 */

#include "CRTMatcher.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

#include "TVector3.h"

#include "larflow/RecoUtils/geofuncs.h"

namespace flashmatch {
namespace dataprep {

// Define static constexpr members
constexpr int CRTMatcher::CRT_TOP_PLANE;
constexpr int CRTMatcher::CRT_BOTTOM_PLANE;
constexpr int CRTMatcher::CRT_FRONT_PLANE;
constexpr int CRTMatcher::CRT_BACK_PLANE;
constexpr int CRTMatcher::CRT_LEFT_PLANE;
constexpr int CRTMatcher::CRT_RIGHT_PLANE;
constexpr double CRTMatcher::US_TO_NS;
constexpr double CRTMatcher::TIME_OFFSET;

CRTMatcher::CRTMatcher(double timing_tolerance, double position_tolerance)
    : _verbosity(1), timing_tolerance_(timing_tolerance), position_tolerance_(position_tolerance),
      total_cosmic_tracks_(0), crt_track_matches_(0), crt_hit_matches_(0),
      total_crt_tracks_(0), total_crt_hits_(0) {
    InitializeCRTGeometry();
}

/**
 * @brief Choose Candidate CRTTracks to match based on coincidence with opflash
 */

int CRTMatcher::FilterCRTTracksByFlashMatches( 
        const std::vector< CRTTrack >& input_crt_tracks, 
        const std::vector< OpticalFlash >&input_opflashes )
{

    _crttrack_index_to_flash_index.clear();

    // Make container for candidate crt tracks
    std::vector< CRTTrack > output_tracks;

    struct RankMatch_t {
        int crt_index;
        int opflash_index;
        double dt;
        bool operator< ( const RankMatch_t& rhs ) {
            if ( dt < rhs.dt )
                return true;
            return false;
        }
    };

    std::vector< RankMatch_t > matchranks_v;

    // loop over given crt tracks
    for (int crt_idx=0; crt_idx<(int)input_crt_tracks.size(); crt_idx++ ) {
        
        auto const& crttrack = input_crt_tracks.at(crt_idx);
        double ave_flashtime = 0.5*( crttrack.startpt_time + crttrack.endpt_time );

        // loop over opflashes and try to match to time
        for (int flash_idx=0; flash_idx<(int)input_opflashes.size(); flash_idx++ ) {
            auto const& opflash = input_opflashes.at(flash_idx);

            // array to hold time difference between both CRT hits
            // that make up the CRT track
            double dt_flashtime = std::fabs( opflash.flash_time - ave_flashtime );
            if ( dt_flashtime<2.0 ) {
                RankMatch_t cand;
                cand.crt_index = crt_idx;
                cand.opflash_index = flash_idx;
                cand.dt = dt_flashtime;
                matchranks_v.push_back( cand );
            }
        }
    }

    std::sort( matchranks_v.begin(), matchranks_v.end() );

    std::cout << "CRT-TRACK to OPFLASH Matches by Time =================" << std::endl;
    for ( auto& cand : matchranks_v ) {
        auto it_map = _crttrack_index_to_flash_index.find( cand.crt_index );
        if ( it_map==_crttrack_index_to_flash_index.end() ) {
            // crt index is not in the map
            // so put in the crt  to opflash index map
            _crttrack_index_to_flash_index[ cand.crt_index ] = cand.opflash_index;
            std::cout << "  CRTTrack[" << cand.crt_index << "]-Opflash[" << cand.opflash_index << "] "
                      << "dt=" << cand.dt << " usec" << std::endl;
        }
    }

    return _crttrack_index_to_flash_index.size();

}

/**
 * @brief match a cosmic track to CRT Track object
 * 
 * CRTTrack objects represent coincident CRT hits in two different panels.
 * They are candidate events for muons passing through one CRT panel,
 * into and out of the TPC, and finally through the other CRT panel.
 * 
 * To check for a match, we ask what fraction of the path through the TPC defined
 * by a line connecting the CRT hits has nearby track hits.
 * 
 * We can do this by binning the path through the TPC. 
 * Then for each ionization hit associated to the track, we can ask which
 * segment along the CRT path it is closest to. With this, we can ask for
 * decision metrics:
 *    - what fraction of the path has nearby hits
 *    - what is the largest gap
 *    - do any of the hits, once you correct for the CRT time relative to the trigger, 
 *      fall outside of the TPC?
 */
int CRTMatcher::MatchToCRTTrack(CRTTrack& crt_track,
                                std::vector<CosmicTrack>& cosmic_tracks,
                                const EventData& input_data,
                                EventData& output_data ) {

    
    int best_match = -1;
    double best_score = -1.0;
    const float step_size = 1.0; // cm  TODO: make configuration parameter
    const float max_dist_to_tpc_path = 10.0; // cm TODO: mke configuration parameter
    
    // Define the path through the CRT

    // We could/should do math to determine intersections with the TPC Wall
    // But we'll do something simpler for now.
    // Lets just step along some fixed step size from one CRThit to another and 
    // record the first and last step inside the TPC.
    // TODO: fix this

    int nsteps = crt_track.length/step_size+1;

    TVector3 firststep_in_tpc;
    TVector3 laststep_in_tpc;
    bool into_tpc  = false;
    bool outof_tpc = false;
    int nsteps_in_tpc = 0;
 
    for (int istep=1; istep<nsteps; istep++) {
        double pathlen = step_size*istep;
        TVector3 pos = crt_track.start_point + crt_track.direction*pathlen;
        if ( pos[0]>=-5.0 && pos[0]<260.0 
            && pos[1]>=-118.0 && pos[1]<118.0
            && pos[2]>=0.0 && pos[2]<1035.0 ) {
            // inside the TPC
            if ( into_tpc == false && outof_tpc==false ) {
                // first step inside the TPC
                into_tpc = true;
                firststep_in_tpc = pos;
            }
            laststep_in_tpc = pos;
            nsteps_in_tpc++;
        }
        else {
            // outside the tpc
            if ( into_tpc==true && outof_tpc==false ) {
                // we've crossed outside
                outof_tpc = true;
                break;
            }
        }
    }

    // if it does not intersect with the TPC, remove it
    if (nsteps_in_tpc==0)
        return -1; // go to next CRTTrack

    float tpc_pathlen = nsteps_in_tpc*step_size;
    std::vector<float> tpc_path_start(3,0);
    std::vector<float> tpc_path_end(3,0);
    std::vector<float> crt_path_dir(3,0);
    for (int i=0; i<3; i++) {
        tpc_path_start[i] = firststep_in_tpc[i];
        crt_path_dir[i]   = crt_track.direction[i];
        tpc_path_end[i]   = laststep_in_tpc[i];
    }

    float crt_ave_time = 0.5*( crt_track.startpt_time + crt_track.endpt_time );
    float x_t0_offset = crt_ave_time*0.109; // (t0 usec x (cm per usec))

    // Now we look for the best match through all the different cosmics
    for ( int icosmic=0; icosmic<(int)cosmic_tracks.size(); icosmic++ ) {

        auto const& cosmic_track = cosmic_tracks.at(icosmic);

        // make the segment counter
        std::vector<int> nhits_per_pathsegment(nsteps_in_tpc, 0);
        int nhits_outside_tpc = 0;

        // now we loop through the track's hits and see what bins along track they fall in.
        for ( size_t ihit=0; ihit<cosmic_track.hitpos_v.size(); ihit++ ) {
            auto const& hitpos = cosmic_track.hitpos_v.at(ihit);

            std::vector<float> correctedpos = hitpos;
            correctedpos[0] -= x_t0_offset;

            // TODO: we can now apply the SCE correction to get the TRUE energy deposit location

            float s = larflow::recoutils::pointRayProjection3f( tpc_path_start, crt_path_dir, correctedpos );
            if ( s>=0 && s<=tpc_pathlen ) {
                // hit falls within the TPC path
                //what is the radius from the TPC path?
                float r = larflow::recoutils::pointLineDistance3f( tpc_path_start, 
                    tpc_path_end, correctedpos );
                if ( r < max_dist_to_tpc_path ) {
                    // close enough to the tpc path
                    // find the bin
                    int ibin = int(s/step_size);
                    nhits_per_pathsegment.at(ibin) += 1;
                }
            }
            else {
                nhits_outside_tpc++;
            }
        }

        // now that we have our pathsegment histogram filled, analyze it
        // we calculate:
        float frac_tpc_path_with_hits = 0.; // number of path segments with a hit
        float max_gaplen = 0.; // length of largest gap between first and last bin with charge
        float front_gaplen = 0.; // length of gap in cm between tpc path start and first segment with a hit
        float back_gaplen = 0.;  // length of gap in cm between tpc path end and last segment with a hit
            
        bool reached_first_segment = false;
        int n_consecutive_above = 0;
        int n_consecutive_zeros = 0;
        int tot_nhits = 0;
        for (int iseg=0; iseg<(int)nhits_per_pathsegment.size(); iseg++) {
            int nhits = nhits_per_pathsegment[iseg];
            if ( nhits==0 ) {
                n_consecutive_above = 0;
                n_consecutive_zeros++;
            }
            else {

                if (  n_consecutive_zeros>=3 ) {
                    // define a gap
                    if ( n_consecutive_zeros>max_gaplen ) {
                        max_gaplen = n_consecutive_zeros;
                    }
                }

                if ( !reached_first_segment && n_consecutive_above==3 ) {
                    front_gaplen = n_consecutive_above-3;
                }

                frac_tpc_path_with_hits += 1.0;

                // reset on versus off counters
                n_consecutive_above++;
                n_consecutive_zeros = 0;
            }
            tot_nhits += nhits;
        }

        if ( n_consecutive_zeros>0 ) {
            back_gaplen = float(n_consecutive_zeros)*step_size;
        }

        if ( nhits_per_pathsegment.size()>0 )
            frac_tpc_path_with_hits /= (float)nhits_per_pathsegment.size();

        float frac_cosmic_hits = float(tot_nhits)/float(cosmic_track.hitpos_v.size());
        float frac_hits_outside_tpc = float(nhits_outside_tpc)/float(cosmic_track.hitpos_v.size());



        // use metrics to determine if the track is a good match to the CRT TPC Path
        bool is_good_match = false;
        if ( frac_cosmic_hits>0.9 && frac_tpc_path_with_hits>0.9 && frac_hits_outside_tpc<0.05 )
            is_good_match = true;

        if ( _verbosity>=kDebug 
              || (_verbosity>=kInfo && is_good_match) ) {
            std::cout << "Cosmic[" << icosmic << "] Results" << std::endl;
            std::cout << "  num steps inside the TPC: " << nsteps_in_tpc << std::endl;
            std::cout << "  nhits outside tpc: " << nhits_outside_tpc << std::endl;
            std::cout << "  fraction of hits outside TPC: " << frac_hits_outside_tpc << std::endl;
            std::cout << "  fraction of path with hits: " << frac_tpc_path_with_hits << std::endl;
            std::cout << "  front gap size: " << front_gaplen << " cm" << std::endl;
            std::cout << "  back gap size: " << back_gaplen << " cm" << std::endl;
            std::cout << "  max gaplen: " << max_gaplen << " cm" << std::endl;
            std::cout << "  num hits close to path: " << tot_nhits << std::endl;
            std::cout << "  frac of cosmic hits close to path: " << frac_cosmic_hits << std::endl;
            if ( is_good_match )
                std::cout << "  ** IS MATCH **" << std::endl;
        }

        // update best match based on fraction of hits close to path
        if (  is_good_match && frac_cosmic_hits>best_score ) {
            best_score = frac_cosmic_hits;
            best_match = icosmic;
        }
        // // Calculate timing difference
        // double time_diff = CalculateTimingDifference(cosmic_track, crt_track);
        // if (time_diff > timing_tolerance_) {
        //     continue; // Skip if timing is incompatible
        // }
        
        // // Calculate spatial distance
        // double spatial_dist = CalculateSpatialDistance(cosmic_track, crt_track);
        // if (spatial_dist > position_tolerance_) {
        //     continue; // Skip if spatially incompatible
        // }

        // // Calculate match score
        // double match_score = CalculateCRTTrackMatchScore(cosmic_track, crt_track);
        
        // if (match_score > best_score) {
        //     best_score = match_score;
        //     best_match = crt_idx;
        // }
    }
    
    if (best_match >= 0) {
        // A match was found. Save it to the event data!

        // but first, find it's opflash match -- or if it has one
        auto it_opflashmap = _crttrack_index_to_flash_index.find( crt_track.index );
        if ( it_opflashmap!=_crttrack_index_to_flash_index.end() ) {

            // this crt track has an opflash match
            auto const& opflash = input_data.optical_flashes.at( it_opflashmap->second );
            output_data.optical_flashes.push_back( opflash );

        }
        else {
            // make an empty flash
            OpticalFlash empty;
            empty.flash_time = crt_ave_time;
            output_data.optical_flashes.emplace_back( std::move(empty) );
        }

        CosmicTrack out_cosmictrack =  cosmic_tracks.at(best_match); // Make a copy
        // shift the x location of the hits, now that we have a t0
        for (auto& hit : out_cosmictrack.hitpos_v ) {
            hit[0] -= x_t0_offset;
            // TODO: apply the Space Charge Effect correction, moving charge to correction position
            // Want a user-friendly utility in larflow::recoutils to do this I think
        }
        // note that the original imgpos are saved -- so we can go back and get the image charge

        output_data.cosmic_tracks.push_back( out_cosmictrack );
        output_data.crt_tracks.push_back( crt_track );

        crt_track_matches_++;
        UpdateStatistics(true, false);
    } else {
        UpdateStatistics(false, false);
    }
    
    return best_match;
}

/**
 * @brief Match cosmic track to crt_hit
 */
std::vector<int> CRTMatcher::MatchToCRTHits(CosmicTrack& cosmic_track,
                                           std::vector<CRTHit>& crt_hits) {

    total_crt_hits_ += crt_hits.size();
    std::vector<int> matched_hits;

    for (size_t hit_idx = 0; hit_idx < crt_hits.size(); ++hit_idx) {
        auto& crt_hit = crt_hits[hit_idx];

        // Check timing compatibility
        if (!IsTimingCompatible(cosmic_track, crt_hit)) {
            continue;
        }

        // Calculate distance from track to hit
        double distance = CalculateTrackToHitDistance(cosmic_track, crt_hit);
        if (distance <= position_tolerance_) {
            matched_hits.push_back(hit_idx);
        }
    }

    if (!matched_hits.empty()) {
        crt_hit_matches_++;
        UpdateStatistics(false, true);
    }

    return matched_hits;
}

double CRTMatcher::CalculateTimingDifference(CosmicTrack& cosmic_track,
                                            CRTTrack& crt_track) {
    // Convert cosmic track time to CRT time reference
    double cosmic_time_ns = ConvertToCRTTime(cosmic_track.anode_crossing_time);

    // Calculate time difference
    double crt_ave_time = 0.5*(crt_track.startpt_time+crt_track.endpt_time);
    return std::abs(crt_ave_time);
}

double CRTMatcher::CalculateSpatialDistance(CosmicTrack& cosmic_track,
                                           CRTTrack& crt_track) {
    // Calculate distance between track endpoints
    double start_dist = (cosmic_track.start_point - crt_track.start_point).Mag();
    double end_dist = (cosmic_track.end_point - crt_track.end_point).Mag();

    // Also check cross distances (start to end, end to start)
    double cross_dist1 = (cosmic_track.start_point - crt_track.end_point).Mag();
    double cross_dist2 = (cosmic_track.end_point - crt_track.start_point).Mag();

    // Return minimum distance
    return std::min({start_dist, end_dist, cross_dist1, cross_dist2});
}

double CRTMatcher::CalculateTrackToHitDistance(CosmicTrack& cosmic_track,
                                              CRTHit& crt_hit) {
    // Calculate minimum distance from track to hit point
    // This is a point-to-line distance calculation

    TVector3 track_vec = cosmic_track.end_point - cosmic_track.start_point;
    TVector3 hit_vec = crt_hit.position - cosmic_track.start_point;

    if (track_vec.Mag() == 0) {
        return hit_vec.Mag(); // Track is a point
    }

    // Project hit onto track line
    double t = hit_vec.Dot(track_vec) / track_vec.Mag2();

    TVector3 closest_point;
    if (t < 0) {
        closest_point = cosmic_track.start_point; // Before track start
    } else if (t > 1) {
        closest_point = cosmic_track.end_point; // After track end
    } else {
        closest_point = cosmic_track.start_point + t * track_vec; // On track
    }

    return (crt_hit.position - closest_point).Mag();
}

bool CRTMatcher::IsTimingCompatible(CosmicTrack& cosmic_track,
                                   CRTHit& crt_hit) {
    double cosmic_time_ns = ConvertToCRTTime(cosmic_track.anode_crossing_time);
    double time_diff = std::abs(cosmic_time_ns - crt_hit.time);

    return time_diff <= timing_tolerance_;
}

double CRTMatcher::CalculateExpectedCRTTime(CosmicTrack& cosmic_track,
                                           int crt_plane_id) {
    // Find intersection of track with CRT plane
    TVector3 intersection = FindCRTIntersection(cosmic_track, crt_plane_id);

    if (intersection.Mag() < 0) {
        return -999.0; // No intersection
    }

    // Calculate time based on track direction and speed
    // This is a simplified calculation
    TVector3 track_vec = cosmic_track.end_point - cosmic_track.start_point;
    TVector3 to_intersection = intersection - cosmic_track.start_point;

    if (track_vec.Mag() > 0) {
        double fraction = to_intersection.Dot(track_vec) / track_vec.Mag2();
        // Assume constant speed along track
        double track_time = fraction * track_vec.Mag() / 30.0; // cm/ns (approximate cosmic speed)
        return ConvertToCRTTime(cosmic_track.anode_crossing_time) + track_time;
    }

    return -999.0;
}

TVector3 CRTMatcher::GetCRTPlanePosition(int plane_id) {
    auto it = crt_planes_.find(plane_id);
    if (it != crt_planes_.end()) {
        return it->second.center;
    }
    return TVector3(-999, -999, -999);
}

TVector3 CRTMatcher::GetCRTPlaneNormal(int plane_id) {
    auto it = crt_planes_.find(plane_id);
    if (it != crt_planes_.end()) {
        return it->second.normal;
    }
    return TVector3(0, 0, 0);
}

TVector3 CRTMatcher::FindCRTIntersection(CosmicTrack& track, int plane_id) {
    auto it = crt_planes_.find(plane_id);
    if (it == crt_planes_.end()) {
        return TVector3(-999, -999, -999); // Invalid plane ID
    }

    CRTPlane& plane = it->second;
    TVector3 track_dir = track.end_point - track.start_point;
    
    if (track_dir.Mag() == 0) {
        return TVector3(-999, -999, -999); // Invalid track
    }

    track_dir = track_dir.Unit();

    // Line-plane intersection
    TVector3 to_plane = plane.center - track.start_point;
    double denominator = track_dir.Dot(plane.normal);

    if (std::abs(denominator) < 1e-6) {
        return TVector3(-999, -999, -999); // Track parallel to plane
    }

    double t = to_plane.Dot(plane.normal) / denominator;
    TVector3 intersection = track.start_point + t * track_dir;

    // Check if intersection is within plane boundaries
    TVector3 to_intersection = intersection - plane.center;
    // Simplified boundary check - assumes rectangular plane aligned with axes
    if (std::abs(to_intersection.Y()) <= plane.height/2.0 && 
        std::abs(to_intersection.Z()) <= plane.width/2.0) {
        return intersection;
    }

    return TVector3(-999, -999, -999); // Outside plane boundaries
}

std::map<std::string, double> CRTMatcher::GetStatistics() {
    std::map<std::string, double> stats;

    stats["total_cosmic_tracks"] = total_cosmic_tracks_;
    stats["crt_track_matches"] = crt_track_matches_;
    stats["crt_hit_matches"] = crt_hit_matches_;
    stats["total_crt_tracks"] = total_crt_tracks_;
    stats["total_crt_hits"] = total_crt_hits_;

    if (total_cosmic_tracks_ > 0) {
        stats["crt_track_efficiency"] = static_cast<double>(crt_track_matches_) / total_cosmic_tracks_;
        stats["crt_hit_efficiency"] = static_cast<double>(crt_hit_matches_) / total_cosmic_tracks_;
    }

    return stats;
}

void CRTMatcher::ResetStatistics() {
    total_cosmic_tracks_ = 0;
    crt_track_matches_ = 0;
    crt_hit_matches_ = 0;
    total_crt_tracks_ = 0;
    total_crt_hits_ = 0;
}

void CRTMatcher::PrintStatistics() {
    auto stats = GetStatistics();

    std::cout << "CRT Matching Statistics:" << std::endl;
    std::cout << "  Total cosmic tracks: " << static_cast<int>(stats["total_cosmic_tracks"]) << std::endl;
    std::cout << "  CRT track matches: " << static_cast<int>(stats["crt_track_matches"]) << std::endl;
    std::cout << "  CRT hit matches: " << static_cast<int>(stats["crt_hit_matches"]) << std::endl;
    std::cout << "  Total CRT tracks: " << static_cast<int>(stats["total_crt_tracks"]) << std::endl;
    std::cout << "  Total CRT hits: " << static_cast<int>(stats["total_crt_hits"]) << std::endl;

    if (stats.find("crt_track_efficiency") != stats.end()) {
        std::cout << "  CRT track efficiency: " << stats["crt_track_efficiency"] * 100.0 << "%" << std::endl;
    }

    if (stats.find("crt_hit_efficiency") != stats.end()) {
        std::cout << "  CRT hit efficiency: " << stats["crt_hit_efficiency"] * 100.0 << "%" << std::endl;
    }
}

double CRTMatcher::CalculateCRTTrackMatchScore(CosmicTrack& cosmic_track,
                                              CRTTrack& crt_track) {
    // Simple scoring based on timing and spatial distance
    double time_diff = CalculateTimingDifference(cosmic_track, crt_track);
    double spatial_dist = CalculateSpatialDistance(cosmic_track, crt_track);

    double time_score = std::exp(-time_diff / timing_tolerance_);
    double spatial_score = std::exp(-spatial_dist / position_tolerance_);

    return time_score * spatial_score;
}

void CRTMatcher::UpdateStatistics(bool crt_track_matched, bool crt_hit_matched) {
    // Statistics are updated in the matching methods
}

void CRTMatcher::InitializeCRTGeometry() {
    // Initialize CRT plane geometry for MicroBooNE
    // These are approximate positions - real implementation would load from geometry service

    // Top plane
    crt_planes_[CRT_TOP_PLANE] = {
        TVector3(128.0, 116.5, 518.0),  // center
        TVector3(0, -1, 0),             // normal (pointing down)
        1000.0,                         // width (Z direction)
        300.0                           // height (X direction)
    };

    // Bottom plane
    crt_planes_[CRT_BOTTOM_PLANE] = {
        TVector3(128.0, -116.5, 518.0), // center
        TVector3(0, 1, 0),              // normal (pointing up)
        1000.0,                         // width (Z direction)
        300.0                           // height (X direction)
    };

    // Front plane (downstream)
    crt_planes_[CRT_FRONT_PLANE] = {
        TVector3(128.0, 0.0, 1036.8),  // center
        TVector3(0, 0, -1),             // normal (pointing upstream)
        300.0,                          // width (X direction)
        233.0                           // height (Y direction)
    };

    // Back plane (upstream)
    crt_planes_[CRT_BACK_PLANE] = {
        TVector3(128.0, 0.0, 0.0),     // center
        TVector3(0, 0, 1),             // normal (pointing downstream)
        300.0,                         // width (X direction)
        233.0                          // height (Y direction)
    };

    // Side planes (if available)
    crt_planes_[CRT_LEFT_PLANE] = {
        TVector3(0.0, 0.0, 518.0),     // center
        TVector3(1, 0, 0),             // normal (pointing right)
        1000.0,                        // width (Z direction)
        233.0                          // height (Y direction)
    };

    crt_planes_[CRT_RIGHT_PLANE] = {
        TVector3(256.4, 0.0, 518.0),   // center
        TVector3(-1, 0, 0),            // normal (pointing left)
        1000.0,                        // width (Z direction)
        233.0                          // height (Y direction)
    };
}

double CRTMatcher::ConvertToCRTTime(double cosmic_time) {
    // Convert from microseconds to nanoseconds and apply any time offset
    return cosmic_time * US_TO_NS + TIME_OFFSET;
}

} // namespace dataprep
} // namespace flashmatch