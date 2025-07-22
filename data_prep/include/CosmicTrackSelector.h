#ifndef COSMICTRACKSELECTOR_H
#define COSMICTRACKSELECTOR_H

#include "DataStructures.h"
#include <string>

namespace flashmatch {
namespace dataprep {

/**
 * @brief Class for applying quality cuts to cosmic ray tracks
 * 
 * This class implements various quality cuts to select clean, well-reconstructed
 * cosmic ray tracks suitable for flash-track matching and neural network training.
 */
class CosmicTrackSelector {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters for quality cuts
     */
    CosmicTrackSelector(QualityCutConfig& config);
    
    /**
     * @brief Destructor
     */
    ~CosmicTrackSelector() = default;
    
    /**
     * @brief Apply all quality cuts to a track
     * @param track The cosmic ray track to evaluate
     * @return true if track passes all cuts, false otherwise
     */
    bool PassesQualityCuts(CosmicTrack& track);
    
    /**
     * @brief Check if track is sufficiently far from detector boundaries
     * @param track The cosmic ray track to evaluate
     * @return true if track passes boundary cuts
     */
    bool PassesBoundaryCuts(CosmicTrack& track);
    
    /**
     * @brief Check track quality metrics (length, hit density, gaps)
     * @param track The cosmic ray track to evaluate
     * @return true if track passes quality cuts
     */
    bool PassesTrackQuality(CosmicTrack& track);
    
    /**
    * @brief Check to see if a track crosses a given plane
    * @param planeCoordStart The plane coordinate of the track start - ie. the one which has a constant value for the whole plane
    * @param planeCoordEnd The plane coordinate of the track end
    * @param firstStart We arbitrarily designate one of the other coordinates as the first. That coordinate's value at the start of the track
    * @param secondStart The first coordinates value at the track end
    * @param plane The value of the plane in the plane coordinate
    * @param firstSlope The slope between the first coordinate and the plane coordinate
    * @param secondSlope The slope between the second coordinate and the plane coordinate
    * @param firstMin The low value of the first coordinate on the plane
    * @param firstMax The high value of the first coordinate on the plane
    * @param secondMin The low value of the second coordinate on the plane
    * @param secondMax The high value of the second coordinate on the plane
    * @param planesCrossed The number of planes crossed by a track, which we pass by reference
    */
    bool CrossesPlane(float planeCoordStart, float planeCoordEnd, float firstStart, 
    float secondStart, float plane, float firstSlope, float secondSlope, 
    float firstMin, float firstMax, float secondMin, float secondMax, int &planesCrossed
    );


    /**
     * @brief Check if track has suitable containment
     * @param track The cosmic ray track to evaluate
     * @return true if track passes containment requirements
     */
    bool PassesContainment(CosmicTrack& track);
    
    /**
     * @brief Update configuration parameters
     * @param config New configuration parameters
     */
    void UpdateConfig(QualityCutConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration parameters
     */
    QualityCutConfig& GetConfig() { return config_; }
    
    /**
     * @brief Load configuration from YAML file
     * @param filename Path to YAML configuration file
     * @return true if successfully loaded
     */
    bool LoadConfigFromFile(std::string& filename);
    
    /**
     * @brief Calculate distance to nearest detector boundary
     * @param point 3D point to evaluate [cm]
     * @return Distance to nearest boundary [cm]
     */
    static double DistanceToBoundary(TVector3& point);
    
    /**
     * @brief Calculate track hit density
     * @param track The cosmic ray track to evaluate
     * @return Hit density [hits/cm]
     */
    static double CalculateHitDensity(CosmicTrack& track);
    
    /**
     * @brief Find largest gap in track
     * @param track The cosmic ray track to evaluate
     * @return Largest gap size [cm]
     */
    static double FindLargestGap(CosmicTrack& track);
    
    /**
     * @brief Get cut statistics
     * @return Map of cut names to pass/fail counts
     */
    std::map<std::string, std::pair<int, int>> GetCutStatistics();
    
    /**
     * @brief Reset cut statistics
     */
    void ResetStatistics();
    
    /**
     * @brief Print cut statistics
     */
    void PrintStatistics();

private:
    QualityCutConfig config_;               ///< Configuration parameters
    
    // Cut statistics (mutable for const methods)
    mutable std::map<std::string, std::pair<int, int>> cut_stats_;
    
    /**
     * @brief Update cut statistics
     * @param cut_name Name of the cut
     * @param passed Whether the cut was passed
     */
    void UpdateStatistics(std::string& cut_name, bool passed);
    
    /**
     * @brief Check if point is within detector fiducial volume
     * @param point 3D point to check [cm]
     * @return true if point is within fiducial volume
     */
    bool IsInFiducialVolume(TVector3& point);
    
    /**
     * @brief Initialize detector geometry parameters
     */
    void InitializeGeometry();
    
    // Detector geometry parameters (MicroBooNE)
    static constexpr double DETECTOR_MIN_X = 0.0;      ///< Detector minimum X [cm]
    static constexpr double DETECTOR_MAX_X = 256.4;    ///< Detector maximum X [cm]
    static constexpr double DETECTOR_MIN_Y = -116.5;   ///< Detector minimum Y [cm]
    static constexpr double DETECTOR_MAX_Y = 116.5;    ///< Detector maximum Y [cm]
    static constexpr double DETECTOR_MIN_Z = 0.0;      ///< Detector minimum Z [cm]
    static constexpr double DETECTOR_MAX_Z = 1036.8;   ///< Detector maximum Z [cm
    static constexpr double DRIFT_VELOCITY = 0.109;    ///< average drift velocity in UB (cm per usec)
};

} // namespace dataprep
} // namespace flashmatch

#endif // COSMICTRACKSELECTOR_H