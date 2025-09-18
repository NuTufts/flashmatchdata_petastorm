/**
 * @file FlashTrackMatcher.cxx
 * @brief Implementation of flash-track matching algorithms
 */

#include "FlashTrackMatcher.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace flashmatch {
namespace dataprep {

// Define static constexpr members
constexpr double FlashTrackMatcher::ANODE_X;
constexpr double FlashTrackMatcher::CATHODE_X;
constexpr int FlashTrackMatcher::NUM_PMTS;

FlashTrackMatcher::FlashTrackMatcher(FlashMatchConfig& config)
    : config_(config), _sce(nullptr) 
{
    // create utility class that lets us go back to the "true" energy deposit location
    // before the space charge effect distortion
    _sce = new larutil::SpaceChargeMicroBooNE( larutil::SpaceChargeMicroBooNE::kMCC9_Backward );    
}

FlashTrackMatcher::~FlashTrackMatcher()
{
    if ( _sce ) {
        delete _sce;
        _sce  = nullptr;
    }
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

        // check image bounds -- no potentiall cut-off tracks
        double xmin_time = bounds[0][0]/config_.drift_velocity;
        double xmax_time = bounds[0][1]/config_.drift_velocity;

        if ( std::fabs(xmax_time-2635) < 20.0 ) {
            continue; // Track is at late image boundary
        }
        if ( std::fabs(xmin_time+400.0) < 20.0 ) {
            continue; // Track is at early image boundary
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
            output_match_data.cosmic_tracks.push_back( out_cosmictrack );

            // make empty crttrack and crthit
            output_match_data.crt_hits.push_back( CRTHit() );
            output_match_data.crt_tracks.push_back( CRTTrack() );

            if ( cand.plane==0 ) {
                output_match_data.match_type.push_back( 0 );
            }
            else if ( cand.plane==1 ) {
                output_match_data.match_type.push_back( 1 );
            }
            else {
                throw std::runtime_error("invalid plane");
            }

            num_matches++;

            break;
        }
    }

    return num_matches;
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

void FlashTrackMatcher::PrintStatistics() {
    // Placeholder for statistics printing
    // Currently no statistics are being tracked
    std::cout << "Flash Matching Statistics:" << std::endl;
    std::cout << "  Statistics tracking not implemented" << std::endl;
}

} // namespace dataprep
} // namespace flashmatch