#ifndef CRTMATCHER_H
#define CRTMATCHER_H

#include "DataStructures.h"
#include <string>

namespace flashmatch {
namespace dataprep {

/**
 * @brief Class for matching cosmic ray tracks with CRT information
 * 
 * This class implements algorithms to associate cosmic ray tracks
 * with CRT (Cosmic Ray Tagger) hits and tracks for improved timing
 * and spatial constraints.
 */
class CRTMatcher {
public:
    /**
     * @brief Constructor
     * @param timing_tolerance Timing tolerance for CRT matching [ns]
     * @param position_tolerance Position tolerance for CRT matching [cm]
     */
    CRTMatcher(double timing_tolerance = 1000.0, double position_tolerance = 30.0);
    
    /**
     * @brief Destructor
     */
    ~CRTMatcher() = default;
    
    /**
     * @brief Match a cosmic ray track to CRT tracks
     * @param cosmic_track The cosmic ray track
     * @param crt_tracks Vector of CRT tracks
     * @return Index of best matching CRT track (-1 if no match)
     */
    int MatchToCRTTrack(const CosmicTrack& cosmic_track,
                       const std::vector<CRTTrack>& crt_tracks) const;
    
    /**
     * @brief Match a cosmic ray track to CRT hits
     * @param cosmic_track The cosmic ray track
     * @param crt_hits Vector of CRT hits
     * @return Vector of indices of matching CRT hits
     */
    std::vector<int> MatchToCRTHits(const CosmicTrack& cosmic_track,
                                   const std::vector<CRTHit>& crt_hits) const;
    
    /**
     * @brief Calculate timing difference between cosmic track and CRT track
     * @param cosmic_track The cosmic ray track
     * @param crt_track The CRT track
     * @return Timing difference [ns]
     */
    double CalculateTimingDifference(const CosmicTrack& cosmic_track,
                                    const CRTTrack& crt_track) const;
    
    /**
     * @brief Calculate spatial distance between cosmic track and CRT track
     * @param cosmic_track The cosmic ray track
     * @param crt_track The CRT track
     * @return Spatial distance [cm]
     */
    double CalculateSpatialDistance(const CosmicTrack& cosmic_track,
                                   const CRTTrack& crt_track) const;
    
    /**
     * @brief Calculate distance from cosmic track to CRT hit
     * @param cosmic_track The cosmic ray track
     * @param crt_hit The CRT hit
     * @return Minimum distance from track to hit [cm]
     */
    double CalculateTrackToHitDistance(const CosmicTrack& cosmic_track,
                                      const CRTHit& crt_hit) const;
    
    /**
     * @brief Check if cosmic track timing is compatible with CRT hit
     * @param cosmic_track The cosmic ray track
     * @param crt_hit The CRT hit
     * @return true if timing is compatible
     */
    bool IsTimingCompatible(const CosmicTrack& cosmic_track,
                           const CRTHit& crt_hit) const;
    
    /**
     * @brief Calculate expected CRT crossing time for cosmic track
     * @param cosmic_track The cosmic ray track
     * @param crt_plane_id CRT plane identifier
     * @return Expected crossing time [ns]
     */
    double CalculateExpectedCRTTime(const CosmicTrack& cosmic_track,
                                   int crt_plane_id) const;
    
    /**
     * @brief Get CRT plane position
     * @param plane_id CRT plane identifier
     * @return 3D position of CRT plane center [cm]
     */
    TVector3 GetCRTPlanePosition(int plane_id) const;
    
    /**
     * @brief Get CRT plane normal vector
     * @param plane_id CRT plane identifier
     * @return Normal vector of CRT plane
     */
    TVector3 GetCRTPlaneNormal(int plane_id) const;
    
    /**
     * @brief Find intersection point of track with CRT plane
     * @param track The cosmic ray track
     * @param plane_id CRT plane identifier
     * @return Intersection point (invalid if no intersection)
     */
    TVector3 FindCRTIntersection(const CosmicTrack& track, int plane_id) const;
    
    /**
     * @brief Set timing tolerance
     * @param tolerance Timing tolerance [ns]
     */
    void SetTimingTolerance(double tolerance) { timing_tolerance_ = tolerance; }
    
    /**
     * @brief Set position tolerance
     * @param tolerance Position tolerance [cm]
     */
    void SetPositionTolerance(double tolerance) { position_tolerance_ = tolerance; }
    
    /**
     * @brief Get timing tolerance
     * @return Timing tolerance [ns]
     */
    double GetTimingTolerance() const { return timing_tolerance_; }
    
    /**
     * @brief Get position tolerance
     * @return Position tolerance [cm]
     */
    double GetPositionTolerance() const { return position_tolerance_; }
    
    /**
     * @brief Get CRT matching statistics
     * @return Map of statistics
     */
    std::map<std::string, double> GetStatistics() const;
    
    /**
     * @brief Reset CRT matching statistics
     */
    void ResetStatistics();
    
    /**
     * @brief Print CRT matching statistics
     */
    void PrintStatistics() const;

private:
    double timing_tolerance_;               ///< Timing tolerance [ns]
    double position_tolerance_;             ///< Position tolerance [cm]
    
    // CRT matching statistics
    mutable int total_cosmic_tracks_;       ///< Total cosmic tracks processed
    mutable int crt_track_matches_;         ///< Number of CRT track matches
    mutable int crt_hit_matches_;           ///< Number of CRT hit matches
    mutable int total_crt_tracks_;          ///< Total CRT tracks processed
    mutable int total_crt_hits_;            ///< Total CRT hits processed
    
    /**
     * @brief Calculate match score for cosmic track and CRT track
     * @param cosmic_track The cosmic ray track
     * @param crt_track The CRT track
     * @return Match score (higher is better)
     */
    double CalculateCRTTrackMatchScore(const CosmicTrack& cosmic_track,
                                      const CRTTrack& crt_track) const;
    
    /**
     * @brief Update CRT matching statistics
     * @param crt_track_matched Whether CRT track match was found
     * @param crt_hit_matched Whether CRT hit matches were found
     */
    void UpdateStatistics(bool crt_track_matched, bool crt_hit_matched) const;
    
    /**
     * @brief Initialize CRT geometry
     */
    void InitializeCRTGeometry();
    
    /**
     * @brief Convert cosmic track time to CRT time reference
     * @param cosmic_time Cosmic track time [μs]
     * @return CRT time [ns]
     */
    double ConvertToCRTTime(double cosmic_time) const;
    
    // CRT geometry parameters (MicroBooNE specific)
    // These would be loaded from geometry service in full implementation
    struct CRTPlane {
        TVector3 center;                    ///< Plane center position [cm]
        TVector3 normal;                    ///< Plane normal vector
        double width;                       ///< Plane width [cm]
        double height;                      ///< Plane height [cm]
    };
    
    std::map<int, CRTPlane> crt_planes_;   ///< CRT plane geometry
    
    // CRT plane IDs (MicroBooNE convention)
    static constexpr int CRT_TOP_PLANE = 0;
    static constexpr int CRT_BOTTOM_PLANE = 1;
    static constexpr int CRT_FRONT_PLANE = 2;
    static constexpr int CRT_BACK_PLANE = 3;
    static constexpr int CRT_LEFT_PLANE = 4;
    static constexpr int CRT_RIGHT_PLANE = 5;
    
    // Time conversion constants
    static constexpr double US_TO_NS = 1000.0;  ///< Microseconds to nanoseconds
    static constexpr double TIME_OFFSET = 0.0;  ///< Time offset between systems [ns]
};

} // namespace dataprep
} // namespace flashmatch

#endif // CRTMATCHER_H