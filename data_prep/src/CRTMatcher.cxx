/**
 * @file CRTMatcher.cxx
 * @brief Implementation of CRT matching algorithms
 */

#include "CRTMatcher.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include "TVector3.h"

#include "larflow/RecoUtils/geofuncs.h"

namespace flashmatch {
namespace dataprep {

// Define static constexpr member
constexpr double CRTMatcher::DRIFT_VELOCITY;

CRTMatcher::CRTMatcher()
    : _verbosity(1), total_cosmic_tracks_(0), crt_track_matches_(0), crt_hit_matches_(0),
      total_crt_tracks_(0), total_crt_hits_(0), _sce(nullptr) {

    // create utility class that lets us go back to the "true" energy deposit location
    // before the space charge effect distortion
    _sce = new larutil::SpaceChargeMicroBooNE( larutil::SpaceChargeMicroBooNE::kMCC9_Backward );
}

CRTMatcher::~CRTMatcher() 
{
    delete _sce;
    _sce = nullptr;
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
 * @brief Choose Candidate CRTTracks to match based on coincidence with opflash
 */
int CRTMatcher::FilterCRTHitsByFlashMatches( 
        const std::vector< CRTHit >& input_crt_hits, 
        const std::vector< OpticalFlash >&input_opflashes )
{

    _crthit_index_to_flash_index.clear();

    // Make container for candidate crt tracks
    std::vector< CRTHit > output_tracks;

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
    for (int crt_idx=0; crt_idx<(int)input_crt_hits.size(); crt_idx++ ) {
        
        auto const& crthit = input_crt_hits.at(crt_idx);

        // loop over opflashes and try to match to time
        for (int flash_idx=0; flash_idx<(int)input_opflashes.size(); flash_idx++ ) {
            auto const& opflash = input_opflashes.at(flash_idx);

            // array to hold time difference between both CRT hits
            // that make up the CRT track
            double dt_flashtime = std::fabs( opflash.flash_time - crthit.time );
            if ( dt_flashtime<2.0 ) {
                RankMatch_t cand;
                cand.crt_index = crthit.index;
                cand.opflash_index = flash_idx;
                cand.dt = dt_flashtime;
                matchranks_v.push_back( cand );
            }
        }
    }

    std::sort( matchranks_v.begin(), matchranks_v.end() );

    std::cout << "CRT-Hit to OPFLASH Matches by Time =================" << std::endl;
    for ( auto& cand : matchranks_v ) {
        auto it_map = _crthit_index_to_flash_index.find( cand.crt_index );
        if ( it_map==_crthit_index_to_flash_index.end() ) {
            // crt index is not in the map
            // so put in the crt  to opflash index map
            _crthit_index_to_flash_index[ cand.crt_index ] = cand.opflash_index;
            std::cout << "  CRTHit[" << cand.crt_index << "]-Opflash[" << cand.opflash_index << "] "
                      << "dt=" << cand.dt << " usec" << std::endl;
        }
    }

    return _crthit_index_to_flash_index.size();

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
    
    // Make dictionary of matches to prevent duplicates
    std::map<int,int> past_matches; // key=cosmic track index, value=opflash index

    for ( size_t imatch=0; imatch<output_data.cosmic_tracks.size(); imatch++) {
        int track_index = output_data.cosmic_tracks.at(imatch).index;
        int flash_index = output_data.optical_flashes.at(imatch).index;
        past_matches[track_index] = flash_index;
    }

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

        // first check if cosmic track is on the image bounds and potentially cut off

        // get the bounds
        double bounds[3][2] = { {1e9,-1e9},{1e9,-1e9},{1e9,-1e9} };

        for (auto const& segpt : cosmic_track.points ) {

            for (int idim=0; idim<3; idim++) {

                if ( segpt[idim]<bounds[idim][0] ) {
                    bounds[idim][0] = segpt[idim];
                }
                if ( segpt[idim]>bounds[idim][1] ) {
                    bounds[idim][1] = segpt[idim];
                }

            }
        }

        // check image bounds -- no potentiall cut-off tracks
        double xmin_time = bounds[0][0]/DRIFT_VELOCITY;
        double xmax_time = bounds[0][1]/DRIFT_VELOCITY;

        if ( std::fabs(xmax_time-2635) < 20.0 ) {
            continue; // Track at late image boundary
        }
        if ( std::fabs(xmin_time+400.0) < 20.0 ) {
            continue; // Track at early image boundary
        }


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
        out_cosmictrack.sce_points.clear();
        for (auto& hit : out_cosmictrack.points ) {
            hit[0] -= x_t0_offset;
            // correct position for space charge effect
            bool applied_sce = false;
            std::vector<double> hit_sce = _sce->ApplySpaceChargeEffect( hit[0], hit[1], hit[2], applied_sce);
            TVector3 hitpos_sce( hit_sce[0], hit_sce[1], hit_sce[2] );
            out_cosmictrack.sce_points.push_back( hitpos_sce );           
        }
        out_cosmictrack.start_point[0] -= x_t0_offset;
        out_cosmictrack.end_point[0]   -= x_t0_offset;
        // note that the original imgpos are saved -- so we can go back and get the image charge

        output_data.cosmic_tracks.push_back( out_cosmictrack );
        output_data.crt_tracks.push_back( crt_track );

        // empty CRT Hit 
        CRTHit empty_crt_hit;
        output_data.crt_hits.push_back( empty_crt_hit );

        output_data.match_type.push_back( 2 );

        crt_track_matches_++;
        UpdateStatistics(true, false);
    } else {
        UpdateStatistics(false, false);
    }
    
    return best_match;
}

/**
 * @brief Match cosmic track to crt_hit
 * 
 * We will begin by defining the direction of the track at the beginning line segments.
 * We will use this to point back to the CRT hit.
 * If close enough, we can also test if there is charge right at the boundary.
 * We will really need the SCE correction to do this properly, lest we limit ourselves
 * to low-X matches where the SCE distortion is limited.
 */
int CRTMatcher::MatchToCRTHits( const CRTHit& crthit, 
    const EventData& input_data, 
    EventData& output_data ) 
{

    int best_match = -1;
    double best_score = -1.0;
    const float step_size = 1.0; // cm  TODO: make configuration parameter
    const float max_dist_to_tpc_path = 10.0; // cm TODO: mke configuration parameter
    const float max_candidate_r = 25.0; // TODO: make configuration parameter

    std::map<int,int> past_matches;
    for ( size_t imatch=0; imatch<output_data.cosmic_tracks.size(); imatch++) {
        int track_index = output_data.cosmic_tracks.at(imatch).index;
        int flash_index = output_data.optical_flashes.at(imatch).index;
        past_matches[track_index] = flash_index;
    }
    
    // the reconstructed position offset coming from the CRT hit time
    float x_t0_offset = crthit.time*0.109; // (t0 usec x (cm per usec))

    std::set<int> matched_tracks;
    std::set<int> matched_flashes;
    for (int imatch=0; imatch<(int)output_data.cosmic_tracks.size(); imatch++ ) {
        matched_tracks.insert( output_data.cosmic_tracks.at(imatch).index );
        matched_flashes.insert( output_data.optical_flashes.at(imatch).index );
    }

    // get the cosmic track container
    auto const& cosmic_tracks = input_data.cosmic_tracks;

    const TVector3& x_crt = crthit.position;

    struct MatchCand_t {
        int itrack;
        int istart;
        double rad;
        TVector3 back_dir;
        MatchCand_t() 
        : itrack(-1), istart(0), rad(1e6)
        {};

        bool operator< ( const MatchCand_t& rhs ) const {
            if ( rad < rhs.rad )
                return true;
            else
                return false;
        };

    };

    std::vector< MatchCand_t > candidates_v;
    double min_r = 1e9;

    // Now we look for the best match through all the different cosmics
    for ( int icosmic=0; icosmic<(int)cosmic_tracks.size(); icosmic++ ) {

        auto const& cosmic_track = cosmic_tracks.at(icosmic);

        // Check if track is on image boundaries
        // get the bounds
        double bounds[3][2] = { {1e9,-1e9},{1e9,-1e9},{1e9,-1e9} };

        for (auto const& segpt : cosmic_track.points ) {

            for (int idim=0; idim<3; idim++) {

                if ( segpt[idim]<bounds[idim][0] ) {
                    bounds[idim][0] = segpt[idim];
                }
                if ( segpt[idim]>bounds[idim][1] ) {
                    bounds[idim][1] = segpt[idim];
                }

            }
        }

        // check image bounds -- no potentiall cut-off tracks
        double xmin_time = bounds[0][0]/DRIFT_VELOCITY;
        double xmax_time = bounds[0][1]/DRIFT_VELOCITY;

        if ( std::fabs(xmax_time-2635) < 20.0 ) {
            continue; // Track at late image boundary
        }
        if ( std::fabs(xmin_time+400.0) < 20.0 ) {
            continue; // Track at early image boundary
        }

        int num_pts = cosmic_track.points.size();

        // determine which end of the track is closest to the the CRT hit position
        // copy so we can make t0 offset correction
        TVector3 startpt = cosmic_track.start_point;
        TVector3 endpt   = cosmic_track.end_point;

        // correct t0 offset
        startpt[0] -= x_t0_offset;
        endpt[0]   -= x_t0_offset;

        double dist2start = (startpt-x_crt).Mag();
        double dist2end   = (endpt-x_crt).Mag();

        int ipt_start = (dist2start<dist2end) ? 0 : num_pts-1;
        int ipt_end   = (dist2start<dist2end) ? num_pts-1 : 0;
        int ipt_icr   = (dist2start<dist2end) ? 1 : -1;

        TVector3 x_track;
        if ( dist2start<dist2end ) {
            x_track = startpt;
        }
        else {
            x_track = endpt;
        }

        // go 10 cm if we can from the end to get the direction of the track end

        double tracklen = 0;
        int ipt = 1;
        TVector3 x_dir;
        while ( tracklen<max_dist_to_tpc_path) {
            TVector3 current_pt = cosmic_track.points.at(ipt_start + ipt_icr*ipt );
            TVector3 last_pt    = cosmic_track.points.at(ipt_start + ipt_icr*(ipt-1) );
            double segment_len = (current_pt-last_pt).Mag();
            if ( tracklen==0 || segment_len+tracklen < max_dist_to_tpc_path ) {
                tracklen += segment_len;
                x_dir = current_pt;
            }
            ipt++;
            if ( ipt==ipt_end || ipt==num_pts )
                break;
        }

        if ( tracklen==0 )
            continue;

        // correct t0 offset for x_dir
        x_dir[0] -= x_t0_offset;

        TVector3 back_dir = (x_track-x_dir).Unit();
        
        // now we get perpendicular distance between CRT hit and track start + backward direction
        std::vector<float> x1     = { static_cast<float>(x_dir[0]), static_cast<float>(x_dir[1]), static_cast<float>(x_dir[2]) };
        std::vector<float> x2     = { static_cast<float>(x_track[0]), static_cast<float>(x_track[1]), static_cast<float>(x_track[2]) };
        std::vector<float> testpt = { static_cast<float>(x_crt[0]), static_cast<float>(x_crt[1]), static_cast<float>(x_crt[2]) };

        // TODO: Do space charge correction for x1 and x2

        double r = larflow::recoutils::pointLineDistance3f( x1, x2, testpt );

        if ( r < max_candidate_r) {

            // std::cout << "register candidate with rad: " << r << std::endl;
            // std::cout << "  dist2start: " << dist2start << std::endl;
            // std::cout << "  dist2end: " << dist2end << std::endl;
            // std::cout << "  cosmic track start: (" << x_track[0] << "," << x_track[1] << "," << x_track[2] << ")" << std::endl;

            // register track as candidate
            MatchCand_t cand;
            cand.itrack = icosmic;
            cand.istart = ipt_start;
            cand.rad = r;
            cand.back_dir = back_dir;
            candidates_v.push_back(cand);
        }

        if ( min_r > r ) {
            min_r = r;
        }

    }
    //std::cout << " number of candidates: " << candidates_v.size() << std::endl;
    //std::cout << " min radius: " << min_r << std::endl;

    if ( candidates_v.size()==0)
        return -1;

    if ( candidates_v.size()>1 ) {
        std::sort( candidates_v.begin(), candidates_v.end() );
    }

    // loop over possible candidates and check entry distance
    for (size_t icand=0; icand < candidates_v.size(); icand++ ) {
        auto const& cand = candidates_v.at(icand);

        // we calculate
        //   1. location where point enters the TPC
        //   2. distance to first charge
        //   3. max gap size (after first charge)
        //   4. num hits outside the TPC
        //   5. fraction of cosmic hits close to the line


        auto const& cosmic_track = cosmic_tracks.at( cand.itrack );

        TVector3 startpt = (cand.istart == 0) ? cosmic_track.start_point : cosmic_track.end_point;
        startpt[0] -= x_t0_offset;
        TVector3 backdir = (startpt-x_crt).Unit();

        int istep=0; 
        TVector3 lastpt = x_crt;
        double tracklen = -1.0;
        TVector3 entrypt = x_crt;
        TVector3 exitpt  = x_crt;
        bool entered_tpc = false;
        bool exited_tpc  = false;

        while ( exited_tpc==false && tracklen<10e3 && istep<10000) {

            tracklen = double(istep)*step_size;
            TVector3 pos = x_crt + (tracklen)*backdir;

            bool intpc = false;
            if ( pos[0]>=-5.0 && pos[0]<260.0 
                && pos[1]>=-118.0 && pos[1]<118.0
                && pos[2]>=0.0 && pos[2]<1036.0 ) {
                // inside the TPC
                intpc = true;
                lastpt = pos;
                if ( !entered_tpc ) {
                    entered_tpc = true;
                    entrypt = pos;
                }
            }

            if ( entered_tpc && !exited_tpc && !intpc ) { 
                exited_tpc = true;
                exitpt = pos;
            }
            istep++;
        }

        double dist2exit  = 1e9;
        double dist2entry = 1e9;

        // now count pts outside the TPC
        int num_out_tpc = 0;
        int num_in_tpc = 0;
        for (auto const& hit : cosmic_track.hitpos_v ) {
            TVector3 pos(hit[0], hit[1], hit[2]);
            pos[0] -= x_t0_offset;

            bool intpc = false; 
            if ( pos[0]>=-5.0 && pos[0]<260.0 
                && pos[1]>=-118.0 && pos[1]<118.0
                && pos[2]>=0.0 && pos[2]<1035.0 ) {
                // inside the TPC
                intpc = true;

                double dexit  = (pos-exitpt).Mag();
                double dentry = (pos-entrypt).Mag();

                if ( dexit < dist2exit ) {
                    dist2exit = dexit;
                }
                if ( dentry < dist2entry ) {
                    dist2entry = dentry;
                }
            }
            if (intpc)
                num_in_tpc++;
            else 
                num_out_tpc++;
        }

        float frac_outside_tpc = num_out_tpc/(num_in_tpc+num_out_tpc);

        // use metrics to determine if the track is a good match to the CRT TPC Path
        bool is_good_match = false;
        if ( frac_outside_tpc<0.10 && num_out_tpc<10 && dist2entry<10.0  ) 
            is_good_match = true;

        if ( _verbosity>=kDebug 
              || (_verbosity>=kInfo && is_good_match) ) {
            std::cout << "CRTHit[" << crthit.index << "]-CosmicTrack[" << cand.itrack << "] Match Candidate Results" << std::endl;
            std::cout << "  cosmic track start: (" << startpt[0] << "," << startpt[1] << "," << startpt[2] << ")" << std::endl;
            std::cout << "  cosmic track dir: (" << backdir[0] << "," << backdir[1] << "," << backdir[2] << ")" << std::endl;
            std::cout << "  candidate.radius: " << cand.rad << " cm" << std::endl;
            std::cout << "  CRT hit pos(with t0 offset): (" << crthit.position[0]+x_t0_offset << "," << crthit.position[1] << "," << crthit.position[2] << ")" << std::endl;
            std::cout << "  CRT hit pos(no t0 offset): (" << crthit.position[0] << "," << crthit.position[1] << "," << crthit.position[2] << ")" << std::endl;
            std::cout << "  num hits inside TPC: " << num_in_tpc << std::endl;
            std::cout << "  num hits outside TPC: " << num_out_tpc << std::endl;
            std::cout << "  fraction of hits outside TPC: " << frac_outside_tpc << std::endl;
            std::cout << "  dist to entry: " << dist2entry << std::endl;
            std::cout << "  dist to exit: " << dist2exit << std::endl;
            if ( is_good_match )
                std::cout << "  ** IS MATCH **" << std::endl;
        }

        if ( is_good_match ) {
            best_match = cand.itrack;
            break;
        }

    }//end of loop over candidates

    
    if (best_match >= 0) {
        // A match was found. Save it to the event data!

        auto it_opflashmap = _crthit_index_to_flash_index.find( crthit.index );

        OpticalFlash cand_match_flash;

        if ( it_opflashmap!=_crthit_index_to_flash_index.end() ) {

            // this crt track has an opflash match
            cand_match_flash = input_data.optical_flashes.at( it_opflashmap->second );
            std::cout << "  CRTHit has matched flash: OpFlash[" << it_opflashmap->second << "]" << std::endl;
            std::cout << "     flash time: " << cand_match_flash.flash_time << std::endl;
            std::cout << "     flash-z: " << cand_match_flash.flash_center[2] << std::endl;
        }
        else {
            std::cout << "  CRT has no matched flash" << std::endl;
        }

        // make sure we are not duplicating a past match
        auto it_pastmatches = past_matches.find( cosmic_tracks.at(best_match).index );
        if ( it_pastmatches!=past_matches.end()) {
            // track already has a match
            // lets see if its to the same flash
            if ( it_pastmatches->second==cand_match_flash.index) {
                // duplicate
                return -1; // no match
            }
        }

        // Match is new: push into output data container
        output_data.optical_flashes.emplace_back( std::move(cand_match_flash) );

        CosmicTrack out_cosmictrack =  cosmic_tracks.at(best_match); // Make a copy
        // shift the x location of the hits, now that we have a t0
        for (auto& hit : out_cosmictrack.hitpos_v ) {
            hit[0] -= x_t0_offset;
            // TODO: apply the Space Charge Effect correction, moving charge to correction position
            // Want a user-friendly utility in larflow::recoutils to do this I think
        }
        out_cosmictrack.sce_points.clear();
        for (auto& hit : out_cosmictrack.points ) {
            hit[0] -= x_t0_offset;
            // correct for the space charge effect
            bool applied_sce = false;
            std::vector<double> hit_sce = _sce->ApplySpaceChargeEffect( hit[0], hit[1], hit[2], applied_sce);
            TVector3 hitpos_sce( hit_sce[0], hit_sce[1], hit_sce[2] );
            out_cosmictrack.sce_points.push_back( hitpos_sce );
        }
        out_cosmictrack.start_point[0] -= x_t0_offset;
        out_cosmictrack.end_point[0]   -= x_t0_offset;
        // note that the original imgpos are saved -- so we can go back and get the image charge

        output_data.cosmic_tracks.push_back( out_cosmictrack );
        output_data.crt_hits.push_back( crthit );
        output_data.crt_tracks.push_back( CRTTrack() ); // empty CRT track to keep alignment

        output_data.match_type.push_back( 3 );

        UpdateStatistics(false, true);
    } else {
        UpdateStatistics(false, false);
    }
    
    return best_match;

}










void CRTMatcher::PrintStatistics() {
    std::cout << "CRT Matching Statistics:" << std::endl;
    std::cout << "  Total cosmic tracks: " << total_cosmic_tracks_ << std::endl;
    std::cout << "  CRT track matches: " << crt_track_matches_ << std::endl;
    std::cout << "  CRT hit matches: " << crt_hit_matches_ << std::endl;
    std::cout << "  Total CRT tracks: " << total_crt_tracks_ << std::endl;
    std::cout << "  Total CRT hits: " << total_crt_hits_ << std::endl;

    if (total_cosmic_tracks_ > 0) {
        double track_eff = static_cast<double>(crt_track_matches_) / total_cosmic_tracks_ * 100.0;
        double hit_eff = static_cast<double>(crt_hit_matches_) / total_cosmic_tracks_ * 100.0;
        std::cout << "  CRT track efficiency: " << track_eff << "%" << std::endl;
        std::cout << "  CRT hit efficiency: " << hit_eff << "%" << std::endl;
    }
}

void CRTMatcher::UpdateStatistics(bool crt_track_matched, bool crt_hit_matched) {
    // Statistics are updated in the matching methods
}

} // namespace dataprep
} // namespace flashmatch