#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include <vector>
#include <map>
#include <string>

// ROOT includes
#include "TVector3.h"

namespace flashmatch {
namespace dataprep {

/**
 * @brief Structure to hold cosmic ray track information
 */
struct CosmicTrack {
    std::vector<TVector3> points;           ///< 3D track points [cm]
    std::vector<double> charge;             ///< Charge deposition at each point [ADC]
    std::vector<double> track_features;     ///< Additional track features (dE/dx, etc.)
    
    double track_length;                    ///< Total track length [cm]
    double total_charge;                    ///< Total charge deposition [ADC]
    TVector3 start_point;                   ///< Track start point [cm]
    TVector3 end_point;                     ///< Track end point [cm]
    TVector3 direction;                     ///< Track direction vector
    
    // Timing information
    double anode_crossing_time;             ///< Expected anode crossing time [μs]
    double cathode_crossing_time;           ///< Expected cathode crossing time [μs]
    
    // Quality metrics
    double hit_density;                     ///< Hits per cm
    double boundary_distance;               ///< Distance to nearest detector boundary [cm]
    bool is_contained;                      ///< Track containment flag
    
    CosmicTrack() : track_length(0), total_charge(0), anode_crossing_time(-999), 
                   cathode_crossing_time(-999), hit_density(0), boundary_distance(0), 
                   is_contained(false) {}
};

/**
 * @brief Structure to hold optical flash information
 */
struct OpticalFlash {
    std::vector<double> pe_per_pmt;         ///< PE count for each PMT (32 PMTs)
    double flash_time;                      ///< Flash time [μs]
    double total_pe;                        ///< Total PE in flash
    TVector3 flash_center;                  ///< Reconstructed flash center [cm]
    double flash_width_y;                   ///< Flash width in Y direction [cm]
    double flash_width_z;                   ///< Flash width in Z direction [cm]
    
    OpticalFlash() : flash_time(-999), total_pe(0), flash_width_y(0), flash_width_z(0) {
        pe_per_pmt.resize(32, 0.0);
    }
};

/**
 * @brief Structure to hold CRT hit information
 */
struct CRTHit {
    TVector3 position;                      ///< CRT hit position [cm]
    double time;                            ///< CRT hit time [ns]
    double charge;                          ///< CRT charge/PE
    int plane_id;                           ///< CRT plane identifier
    int strip_id;                           ///< CRT strip identifier
    
    CRTHit() : time(-999), charge(0), plane_id(-1), strip_id(-1) {}
};

/**
 * @brief Structure to hold CRT track information
 */
struct CRTTrack {
    TVector3 start_point;                   ///< CRT track start [cm]
    TVector3 end_point;                     ///< CRT track end [cm]
    TVector3 direction;                     ///< CRT track direction
    double time;                            ///< CRT track time [ns]
    double length;                          ///< CRT track length [cm]
    std::vector<int> associated_hits;       ///< Indices of associated CRT hits
    
    CRTTrack() : time(-999), length(0) {}
};

/**
 * @brief Structure to hold flash-track match information
 */
struct FlashTrackMatch {
    int track_id;                           ///< Index of matched track
    int flash_id;                           ///< Index of matched flash
    
    // Matching metrics
    double time_difference;                 ///< Time difference [μs]
    double spatial_distance;                ///< Spatial distance [cm]
    double pe_prediction_residual;          ///< Prediction vs actual PE difference
    double match_score;                     ///< Overall match quality score
    
    // CRT information (if available)
    bool has_crt_match;                     ///< CRT match available
    int crt_track_id;                       ///< Index of matched CRT track
    std::vector<int> crt_hit_ids;           ///< Indices of matched CRT hits
    double crt_time_difference;             ///< CRT time difference [ns]
    
    FlashTrackMatch() : track_id(-1), flash_id(-1), time_difference(999), 
                       spatial_distance(999), pe_prediction_residual(999), 
                       match_score(0), has_crt_match(false), crt_track_id(-1),
                       crt_time_difference(999) {}
};

/**
 * @brief Event-level data container
 */
struct EventData {
    // Event identification
    int run;
    int subrun;
    int event;
    
    // Data collections
    std::vector<CosmicTrack> cosmic_tracks;
    std::vector<OpticalFlash> optical_flashes;
    std::vector<CRTHit> crt_hits;
    std::vector<CRTTrack> crt_tracks;
    std::vector<FlashTrackMatch> flash_track_matches;
    
    // Event-level quality metrics
    int num_quality_tracks;                 ///< Number of tracks passing quality cuts
    int num_matched_flashes;                ///< Number of successfully matched flashes
    double event_charge;                    ///< Total event charge [ADC]
    double event_pe;                        ///< Total event PE
    
    EventData() : run(-1), subrun(-1), event(-1), num_quality_tracks(0), 
                 num_matched_flashes(0), event_charge(0), event_pe(0) {}
};

/**
 * @brief Configuration parameters for quality cuts
 */
struct QualityCutConfig {
    // Boundary cuts
    double min_distance_to_edge;            ///< Minimum distance to detector edge [cm]
    bool require_both_ends_contained;       ///< Require both track ends contained
    
    // Track quality
    double min_track_length;                ///< Minimum track length [cm]
    double min_hit_density;                 ///< Minimum hits per cm
    double max_gap_size;                    ///< Maximum gap in track [cm]
    
    // Flash requirements
    double timing_window;                   ///< Timing window for flash matching [μs]
    double pe_threshold;                    ///< Minimum PE in flash
    
    // CRT requirements
    double crt_timing_tolerance;            ///< CRT timing tolerance [μs]
    double crt_position_tolerance;          ///< CRT position tolerance [cm]
    
    QualityCutConfig() : min_distance_to_edge(10.0), require_both_ends_contained(false),
                        min_track_length(50.0), min_hit_density(0.5), max_gap_size(5.0),
                        timing_window(23.4), pe_threshold(50.0), 
                        crt_timing_tolerance(1.0), crt_position_tolerance(30.0) {}
};

/**
 * @brief Configuration parameters for flash-track matching
 */
struct FlashMatchConfig {
    // Timing matching
    double anode_crossing_tolerance;        ///< Anode crossing time tolerance [μs]
    double cathode_crossing_tolerance;      ///< Cathode crossing time tolerance [μs]
    double drift_velocity;                  ///< Drift velocity [cm/μs]
    
    // Spatial matching
    double track_flash_distance_cut;        ///< Maximum track-flash distance [cm]
    double pmt_coverage_requirement;        ///< Minimum PMT coverage fraction
    
    // CRT integration
    bool enable_crt_track_matching;         ///< Enable CRT track matching
    bool enable_crt_hit_matching;           ///< Enable CRT hit matching
    double crt_timing_precision;            ///< CRT timing precision [ns]
    
    FlashMatchConfig() : anode_crossing_tolerance(0.5), cathode_crossing_tolerance(0.5),
                        drift_velocity(0.1098), track_flash_distance_cut(100.0),
                        pmt_coverage_requirement(0.3), enable_crt_track_matching(true),
                        enable_crt_hit_matching(true), crt_timing_precision(1.0) {}
};

} // namespace dataprep
} // namespace flashmatch

#endif // DATASTRUCTURES_H