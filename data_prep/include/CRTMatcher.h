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

    std::vector< CRTTrack > FilterCRTTracksByFlashMatches( 
        const std::vector< CRTTrack >& input_crt_tracks, 
        const std::vector< OpticalFlash >&input_opflashes );
    
    /**
     * @brief Match a cosmic ray track to CRT tracks
     * @param cosmic_track The cosmic ray track
     * @param crt_tracks Vector of CRT tracks
     * @return Index of best matching CRT track (-1 if no match)
     */
    int MatchToCRTTrack(CosmicTrack& cosmic_track,
                       std::vector<CRTTrack>& crt_tracks);
    
    /**
     * @brief Match a cosmic ray track to CRT hits
     * @param cosmic_track The cosmic ray track
     * @param crt_hits Vector of CRT hits
     * @return Vector of indices of matching CRT hits
     */
    std::vector<int> MatchToCRTHits(CosmicTrack& cosmic_track,
                                   std::vector<CRTHit>& crt_hits);
    
    /**
     * @brief Calculate timing difference between cosmic track and CRT track
     * @param cosmic_track The cosmic ray track
     * @param crt_track The CRT track
     * @return Timing difference [ns]
     */
    double CalculateTimingDifference(CosmicTrack& cosmic_track,
                                    CRTTrack& crt_track);
    
    /**
     * @brief Calculate spatial distance between cosmic track and CRT track
     * @param cosmic_track The cosmic ray track
     * @param crt_track The CRT track
     * @return Spatial distance [cm]
     */
    double CalculateSpatialDistance(CosmicTrack& cosmic_track,
                                   CRTTrack& crt_track);
    
    /**
     * @brief Calculate distance from cosmic track to CRT hit
     * @param cosmic_track The cosmic ray track
     * @param crt_hit The CRT hit
     * @return Minimum distance from track to hit [cm]
     */
    double CalculateTrackToHitDistance(CosmicTrack& cosmic_track,
                                      CRTHit& crt_hit);
    
    /**
     * @brief Check if cosmic track timing is compatible with CRT hit
     * @param cosmic_track The cosmic ray track
     * @param crt_hit The CRT hit
     * @return true if timing is compatible
     */
    bool IsTimingCompatible(CosmicTrack& cosmic_track,
                           CRTHit& crt_hit);
    
    /**
     * @brief Calculate expected CRT crossing time for cosmic track
     * @param cosmic_track The cosmic ray track
     * @param crt_plane_id CRT plane identifier
     * @return Expected crossing time [ns]
     */
    double CalculateExpectedCRTTime(CosmicTrack& cosmic_track,
                                   int crt_plane_id);
    
    /**
     * @brief Get CRT plane position
     * @param plane_id CRT plane identifier
     * @return 3D position of CRT plane center [cm]
     */
    TVector3 GetCRTPlanePosition(int plane_id);
    
    /**
     * @brief Get CRT plane normal vector
     * @param plane_id CRT plane identifier
     * @return Normal vector of CRT plane
     */
    TVector3 GetCRTPlaneNormal(int plane_id);
    
    /**
     * @brief Find intersection point of track with CRT plane
     * @param track The cosmic ray track
     * @param plane_id CRT plane identifier
     * @return Intersection point (invalid if no intersection)
     */
    TVector3 FindCRTIntersection(CosmicTrack& track, int plane_id);
    
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
    double GetTimingTolerance() { return timing_tolerance_; }
    
    /**
     * @brief Get position tolerance
     * @return Position tolerance [cm]
     */
    double GetPositionTolerance() { return position_tolerance_; }
    
    /**
     * @brief Get CRT matching statistics
     * @return Map of statistics
     */
    std::map<std::string, double> GetStatistics();
    
    /**
     * @brief Reset CRT matching statistics
     */
    void ResetStatistics();
    
    /**
     * @brief Print CRT matching statistics
     */
    void PrintStatistics();

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
    double CalculateCRTTrackMatchScore(CosmicTrack& cosmic_track,
                                      CRTTrack& crt_track);
    
    /**
     * @brief Update CRT matching statistics
     * @param crt_track_matched Whether CRT track match was found
     * @param crt_hit_matched Whether CRT hit matches were found
     */
    void UpdateStatistics(bool crt_track_matched, bool crt_hit_matched);
    
    /**
     * @brief Initialize CRT geometry
     */
    void InitializeCRTGeometry();
    
    /**
     * @brief Convert cosmic track time to CRT time reference
     * @param cosmic_time Cosmic track time [μs]
     * @return CRT time [ns]
     */
    double ConvertToCRTTime(double cosmic_time);
    
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