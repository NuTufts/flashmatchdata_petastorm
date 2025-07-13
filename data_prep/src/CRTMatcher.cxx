/**
 * @file CRTMatcher.cxx
 * @brief Implementation of CRT matching algorithms
 */

#include "CRTMatcher.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

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
    : timing_tolerance_(timing_tolerance), position_tolerance_(position_tolerance),
      total_cosmic_tracks_(0), crt_track_matches_(0), crt_hit_matches_(0),
      total_crt_tracks_(0), total_crt_hits_(0) {
    InitializeCRTGeometry();
}

int CRTMatcher::MatchToCRTTrack(const CosmicTrack& cosmic_track,
                               const std::vector<CRTTrack>& crt_tracks) const {
    
    total_cosmic_tracks_++;
    total_crt_tracks_ += crt_tracks.size();
    
    int best_match = -1;
    double best_score = -1.0;
    
    for (size_t crt_idx = 0; crt_idx < crt_tracks.size(); ++crt_idx) {
        const auto& crt_track = crt_tracks[crt_idx];
        
        // Calculate timing difference
        double time_diff = CalculateTimingDifference(cosmic_track, crt_track);
        if (time_diff > timing_tolerance_) {
            continue; // Skip if timing is incompatible
        }
        
        // Calculate spatial distance
        double spatial_dist = CalculateSpatialDistance(cosmic_track, crt_track);
        if (spatial_dist > position_tolerance_) {
            continue; // Skip if spatially incompatible
        }
        
        // Calculate match score
        double match_score = CalculateCRTTrackMatchScore(cosmic_track, crt_track);
        
        if (match_score > best_score) {
            best_score = match_score;
            best_match = crt_idx;
        }
    }
    
    if (best_match >= 0) {
        crt_track_matches_++;
        UpdateStatistics(true, false);
    } else {
        UpdateStatistics(false, false);
    }
    
    return best_match;
}

std::vector<int> CRTMatcher::MatchToCRTHits(const CosmicTrack& cosmic_track,
                                           const std::vector<CRTHit>& crt_hits) const {
    
    total_crt_hits_ += crt_hits.size();
    std::vector<int> matched_hits;
    
    for (size_t hit_idx = 0; hit_idx < crt_hits.size(); ++hit_idx) {
        const auto& crt_hit = crt_hits[hit_idx];
        
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

double CRTMatcher::CalculateTimingDifference(const CosmicTrack& cosmic_track,
                                            const CRTTrack& crt_track) const {
    // Convert cosmic track time to CRT time reference
    double cosmic_time_ns = ConvertToCRTTime(cosmic_track.anode_crossing_time);
    
    // Calculate time difference
    return std::abs(cosmic_time_ns - crt_track.time);
}

double CRTMatcher::CalculateSpatialDistance(const CosmicTrack& cosmic_track,
                                           const CRTTrack& crt_track) const {
    // Calculate distance between track endpoints
    double start_dist = (cosmic_track.start_point - crt_track.start_point).Mag();
    double end_dist = (cosmic_track.end_point - crt_track.end_point).Mag();
    
    // Also check cross distances (start to end, end to start)
    double cross_dist1 = (cosmic_track.start_point - crt_track.end_point).Mag();
    double cross_dist2 = (cosmic_track.end_point - crt_track.start_point).Mag();
    
    // Return minimum distance
    return std::min({start_dist, end_dist, cross_dist1, cross_dist2});
}

double CRTMatcher::CalculateTrackToHitDistance(const CosmicTrack& cosmic_track,
                                              const CRTHit& crt_hit) const {
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

bool CRTMatcher::IsTimingCompatible(const CosmicTrack& cosmic_track,
                                   const CRTHit& crt_hit) const {
    double cosmic_time_ns = ConvertToCRTTime(cosmic_track.anode_crossing_time);
    double time_diff = std::abs(cosmic_time_ns - crt_hit.time);
    
    return time_diff <= timing_tolerance_;
}

double CRTMatcher::CalculateExpectedCRTTime(const CosmicTrack& cosmic_track,
                                           int crt_plane_id) const {
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

TVector3 CRTMatcher::GetCRTPlanePosition(int plane_id) const {
    auto it = crt_planes_.find(plane_id);
    if (it != crt_planes_.end()) {
        return it->second.center;
    }
    return TVector3(-999, -999, -999);
}

TVector3 CRTMatcher::GetCRTPlaneNormal(int plane_id) const {
    auto it = crt_planes_.find(plane_id);
    if (it != crt_planes_.end()) {
        return it->second.normal;
    }
    return TVector3(0, 0, 0);
}

TVector3 CRTMatcher::FindCRTIntersection(const CosmicTrack& track, int plane_id) const {
    auto it = crt_planes_.find(plane_id);
    if (it == crt_planes_.end()) {
        return TVector3(-999, -999, -999); // Invalid plane ID
    }
    
    const CRTPlane& plane = it->second;
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

std::map<std::string, double> CRTMatcher::GetStatistics() const {
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

void CRTMatcher::PrintStatistics() const {
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

double CRTMatcher::CalculateCRTTrackMatchScore(const CosmicTrack& cosmic_track,
                                              const CRTTrack& crt_track) const {
    // Simple scoring based on timing and spatial distance
    double time_diff = CalculateTimingDifference(cosmic_track, crt_track);
    double spatial_dist = CalculateSpatialDistance(cosmic_track, crt_track);
    
    double time_score = std::exp(-time_diff / timing_tolerance_);
    double spatial_score = std::exp(-spatial_dist / position_tolerance_);
    
    return time_score * spatial_score;
}

void CRTMatcher::UpdateStatistics(bool crt_track_matched, bool crt_hit_matched) const {
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

double CRTMatcher::ConvertToCRTTime(double cosmic_time) const {
    // Convert from microseconds to nanoseconds and apply any time offset
    return cosmic_time * US_TO_NS + TIME_OFFSET;
}

} // namespace dataprep
} // namespace flashmatch