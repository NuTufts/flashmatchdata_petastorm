/**
 * @file CosmicTrackSelector.cxx
 * @brief Implementation of cosmic ray track quality selection
 */

#include "CosmicTrackSelector.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace flashmatch {
namespace dataprep {

// Define static constexpr members
constexpr double CosmicTrackSelector::DETECTOR_MIN_X;
constexpr double CosmicTrackSelector::DETECTOR_MAX_X;
constexpr double CosmicTrackSelector::DETECTOR_MIN_Y;
constexpr double CosmicTrackSelector::DETECTOR_MAX_Y;
constexpr double CosmicTrackSelector::DETECTOR_MIN_Z;
constexpr double CosmicTrackSelector::DETECTOR_MAX_Z;
constexpr double DRIFT_VELOCITY = 0.109;    ///< average drift velocity in UB (cm per usec)

CosmicTrackSelector::CosmicTrackSelector(QualityCutConfig& config)
    : config_(config) {
    InitializeGeometry();
}

bool CosmicTrackSelector::PassesQualityCuts(CosmicTrack& track) {
    bool passes_boundary = PassesBoundaryCuts(track);
    bool passes_quality = PassesTrackQuality(track);
    bool passes_containment = PassesContainment(track);

    std::string boundaryCutString = "boundary_cuts";
    std::string trackQualityString = "track_quality";
    std::string containmentString = "containment";

    UpdateStatistics(boundaryCutString, passes_boundary);
    UpdateStatistics(trackQualityString, passes_quality);
    UpdateStatistics(containmentString, passes_containment);

    std::string allCutsString = "all_cuts";
    
    bool passes_all = passes_boundary && passes_quality && passes_containment;
    UpdateStatistics(allCutsString, passes_all);

    return passes_all;
}

bool CosmicTrackSelector::PassesBoundaryCuts(CosmicTrack& track) {
    // Check minimum distance to detector edge
    if (track.boundary_distance < config_.min_distance_to_edge) {
        return false;
    }
    
    // Check containment requirements if specified
    if (config_.require_both_ends_contained) {
        bool start_contained = IsInFiducialVolume(track.start_point);
        bool end_contained = IsInFiducialVolume(track.end_point);
        return start_contained && end_contained;
    }
    
    return true;
}

bool CosmicTrackSelector::PassesTrackQuality(CosmicTrack& track) {
    // Check minimum track length
    if (track.track_length < config_.min_track_length) {
        return false;
    }

    // Check hit density
    double hit_density = CalculateHitDensity(track);
    if (hit_density < config_.min_hit_density) {
        return false;
    }

    // Check maximum gap size
    double largest_gap = FindLargestGap(track);
    if (largest_gap > config_.max_gap_size) {
        return false;
    }

    return true;
}

bool CosmicTrackSelector::CrossesPlane(float planeCoordStart, float planeCoordEnd, float firstStart, 
    float secondStart, float plane, float firstSlope, float secondSlope, 
    float firstMin, float firstMax, float secondMin, float secondMax, int &crossedPlanes
    ) {
    // First see if the particle crosses the plane's dimension
    if ((planeCoordStart < plane and planeCoordEnd > plane) or (planeCoordStart > plane and planeCoordEnd < plane)) {
        // Now that we know it does, find the point at which it crosses
        float firstCoord = firstSlope*(plane - planeCoordStart) + firstStart;
        float secondCoord = secondSlope*(plane - planeCoordStart) + secondStart;

        // See if that point is on the detector plane, or if it passes by it
        if (firstCoord >= firstMin
            and firstCoord <= firstMax
            and secondCoord >= secondMin
            and secondCoord <= secondMax
        ) {
            crossedPlanes ++;
            return true; // True means we know the point crosses
        }
        // If it either didn't fall on the plane or didn't cross the dimension, we return false
        else {
            return false;
        }
    }

    else {
        return false;
    }
}

bool CosmicTrackSelector::PassesContainment(CosmicTrack& track) {
    // Basic containment check - at least one end should be in fiducial volume
    // or track should cross the detector

    bool start_contained = IsInFiducialVolume(track.start_point);
    bool end_contained = IsInFiducialVolume(track.end_point);

    int planesCrossed = 0;
    
    // If both ends are outside, track might still be valid if it crosses
    if (!start_contained || !end_contained) {
        // Check if track crosses detector edge planes - simplified check
        //bool crosses_x_plane = (track.start_point.X() < DETECTOR_MIN_X && track.end_point.X() > DETECTOR_MAX_X) ||
        //                (track.start_point.X() > DETECTOR_MAX_X && track.end_point.X() < DETECTOR_MIN_X);
        //bool crosses_y_plane = (track.start_point.Y() < DETECTOR_MIN_Y && track.end_point.Y() > DETECTOR_MAX_Y) ||
        //                (track.start_point.Y() > DETECTOR_MAX_Y && track.end_point.Y() < DETECTOR_MIN_Y);
        //bool crosses_z_plane = (track.start_point.Z() < DETECTOR_MIN_Z && track.end_point.Z() > DETECTOR_MAX_Z) ||
        //                (track.start_point.Z() > DETECTOR_MAX_Z && track.end_point.Z() < DETECTOR_MIN_Z);

        // Check to see if the track actually crosses each plane
        double slope_xy = (track.start_point.Y() - track.end_point.Y())/(track.start_point.X() - track.end_point.X());
        double slope_xz = (track.start_point.Z() - track.end_point.Z())/(track.start_point.X() - track.end_point.X());
        double slope_yx = (track.start_point.X() - track.end_point.X())/(track.start_point.Y() - track.end_point.Y());
        double slope_yz = (track.start_point.Z() - track.end_point.Z())/(track.start_point.Y() - track.end_point.Y());
        double slope_zx = (track.start_point.X() - track.end_point.X())/(track.start_point.Z() - track.end_point.Z());
        double slope_zy = (track.start_point.Y() - track.end_point.Y())/(track.start_point.Z() - track.end_point.Z());

        bool crossesCathode = CrossesPlane(track.start_point.X(), track.end_point.X(), track.start_point.Y(), track.start_point.Z(), DETECTOR_MAX_X, slope_xy, slope_xz, DETECTOR_MIN_Y, DETECTOR_MAX_Y, DETECTOR_MIN_Z, DETECTOR_MAX_Z, planesCrossed);
        bool crossesAnode = CrossesPlane(track.start_point.X(), track.end_point.X(), track.start_point.Y(), track.start_point.Z(), DETECTOR_MAX_X, slope_xy, slope_xz, DETECTOR_MIN_Y, DETECTOR_MAX_Y, DETECTOR_MIN_Z, DETECTOR_MAX_Z, planesCrossed);
        bool crossesFirstY = CrossesPlane(track.start_point.X(), track.end_point.X(), track.start_point.Y(), track.start_point.Z(), DETECTOR_MAX_X, slope_xy, slope_xz, DETECTOR_MIN_Y, DETECTOR_MAX_Y, DETECTOR_MIN_Z, DETECTOR_MAX_Z, planesCrossed);
        bool crossesSecondY = CrossesPlane(track.start_point.X(), track.end_point.X(), track.start_point.Y(), track.start_point.Z(), DETECTOR_MAX_X, slope_xy, slope_xz, DETECTOR_MIN_Y, DETECTOR_MAX_Y, DETECTOR_MIN_Z, DETECTOR_MAX_Z, planesCrossed);
        bool crossesFirstZ = CrossesPlane(track.start_point.X(), track.end_point.X(), track.start_point.Y(), track.start_point.Z(), DETECTOR_MAX_X, slope_xy, slope_xz, DETECTOR_MIN_Y, DETECTOR_MAX_Y, DETECTOR_MIN_Z, DETECTOR_MAX_Z, planesCrossed);
        bool crossesSecondZ = CrossesPlane(track.start_point.X(), track.end_point.X(), track.start_point.Y(), track.start_point.Z(), DETECTOR_MAX_X, slope_xy, slope_xz, DETECTOR_MIN_Y, DETECTOR_MAX_Y, DETECTOR_MIN_Z, DETECTOR_MAX_Z, planesCrossed);
        }

    return planesCrossed;
}

void CosmicTrackSelector::UpdateConfig(QualityCutConfig& config) {
    config_ = config;
}

bool CosmicTrackSelector::LoadConfigFromFile(std::string& filename) {
    // TODO: Implement YAML configuration loading
    // For now, just print that we would load from file
    std::cout << "Loading quality cuts configuration from: " << filename << std::endl;
    
    // Check if file exists
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open configuration file: " << filename << std::endl;
        return false;
    }
    
    // TODO: Parse YAML and update config_
    // This would require a YAML parsing library like yaml-cpp
    
    return true;
}

double CosmicTrackSelector::DistanceToBoundary(TVector3& point) {
    double dist_x_min = point.X() - DETECTOR_MIN_X;
    double dist_x_max = DETECTOR_MAX_X - point.X();
    double dist_y_min = point.Y() - DETECTOR_MIN_Y;
    double dist_y_max = DETECTOR_MAX_Y - point.Y();
    double dist_z_min = point.Z() - DETECTOR_MIN_Z;
    double dist_z_max = DETECTOR_MAX_Z - point.Z();
    
    double min_dist = std::min({dist_x_min, dist_x_max, dist_y_min, 
                               dist_y_max, dist_z_min, dist_z_max});
    
    return min_dist;
}

double CosmicTrackSelector::CalculateHitDensity(CosmicTrack& track) {
    if (track.track_length <= 0) return 0.0;
    return static_cast<double>(track.points.size()) / track.track_length;
}

double CosmicTrackSelector::FindLargestGap(CosmicTrack& track) {
    if (track.points.size() < 2) return 0.0;
    
    double largest_gap = 0.0;
    for (size_t i = 1; i < track.points.size(); ++i) {
        double gap = (track.points[i] - track.points[i-1]).Mag();
        largest_gap = std::max(largest_gap, gap);
    }
    
    return largest_gap;
}

std::map<std::string, std::pair<int, int>> CosmicTrackSelector::GetCutStatistics() {
    return cut_stats_;
}

void CosmicTrackSelector::ResetStatistics() {
    cut_stats_.clear();
}

void CosmicTrackSelector::PrintStatistics() {
    std::cout << "Quality Cut Statistics:" << std::endl;
    for (auto& stat : cut_stats_) {
        int passed = stat.second.first;
        int total = stat.second.first + stat.second.second;
        double efficiency = total > 0 ? static_cast<double>(passed) / total * 100.0 : 0.0;
        
        std::cout << "  " << stat.first << ": " << passed << "/" << total 
                  << " (" << efficiency << "%)" << std::endl;
    }
}

void CosmicTrackSelector::UpdateStatistics(std::string& cut_name, bool passed) {
    if (cut_stats_.find(cut_name) == cut_stats_.end()) {
        cut_stats_[cut_name] = std::make_pair(0, 0);
    }
    
    if (passed) {
        cut_stats_[cut_name].first++;
    } else {
        cut_stats_[cut_name].second++;
    }
}

bool CosmicTrackSelector::IsInFiducialVolume(TVector3& point) {
    double dist_to_boundary = DistanceToBoundary(point);
    return dist_to_boundary >= config_.min_distance_to_edge;
}

void CosmicTrackSelector::InitializeGeometry() {
    // Geometry initialization would go here
    // For now, detector boundaries are defined as static constants
}

} // namespace dataprep
} // namespace flashmatch