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
    CosmicTrackSelector(const QualityCutConfig& config);
    
    /**
     * @brief Destructor
     */
    ~CosmicTrackSelector() = default;
    
    /**
     * @brief Apply all quality cuts to a track
     * @param track The cosmic ray track to evaluate
     * @return true if track passes all cuts, false otherwise
     */
    bool PassesQualityCuts(const CosmicTrack& track) const;
    
    /**
     * @brief Check if track is sufficiently far from detector boundaries
     * @param track The cosmic ray track to evaluate
     * @return true if track passes boundary cuts
     */
    bool PassesBoundaryCuts(const CosmicTrack& track) const;
    
    /**
     * @brief Check track quality metrics (length, hit density, gaps)
     * @param track The cosmic ray track to evaluate
     * @return true if track passes quality cuts
     */
    bool PassesTrackQuality(const CosmicTrack& track) const;
    
    /**
     * @brief Check if track has suitable containment
     * @param track The cosmic ray track to evaluate
     * @return true if track passes containment requirements
     */
    bool PassesContainment(const CosmicTrack& track) const;
    
    /**
     * @brief Update configuration parameters
     * @param config New configuration parameters
     */
    void UpdateConfig(const QualityCutConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration parameters
     */
    const QualityCutConfig& GetConfig() const { return config_; }
    
    /**
     * @brief Load configuration from YAML file
     * @param filename Path to YAML configuration file
     * @return true if successfully loaded
     */
    bool LoadConfigFromFile(const std::string& filename);
    
    /**
     * @brief Calculate distance to nearest detector boundary
     * @param point 3D point to evaluate [cm]
     * @return Distance to nearest boundary [cm]
     */
    static double DistanceToBoundary(const TVector3& point);
    
    /**
     * @brief Calculate track hit density
     * @param track The cosmic ray track to evaluate
     * @return Hit density [hits/cm]
     */
    static double CalculateHitDensity(const CosmicTrack& track);
    
    /**
     * @brief Find largest gap in track
     * @param track The cosmic ray track to evaluate
     * @return Largest gap size [cm]
     */
    static double FindLargestGap(const CosmicTrack& track);
    
    /**
     * @brief Get cut statistics
     * @return Map of cut names to pass/fail counts
     */
    std::map<std::string, std::pair<int, int>> GetCutStatistics() const;
    
    /**
     * @brief Reset cut statistics
     */
    void ResetStatistics();
    
    /**
     * @brief Print cut statistics
     */
    void PrintStatistics() const;

private:
    QualityCutConfig config_;               ///< Configuration parameters
    
    // Cut statistics (mutable for const methods)
    mutable std::map<std::string, std::pair<int, int>> cut_stats_;
    
    /**
     * @brief Update cut statistics
     * @param cut_name Name of the cut
     * @param passed Whether the cut was passed
     */
    void UpdateStatistics(const std::string& cut_name, bool passed) const;
    
    /**
     * @brief Check if point is within detector fiducial volume
     * @param point 3D point to check [cm]
     * @return true if point is within fiducial volume
     */
    bool IsInFiducialVolume(const TVector3& point) const;
    
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
    static constexpr double DETECTOR_MAX_Z = 1036.8;   ///< Detector maximum Z [cm]
};

} // namespace dataprep
} // namespace flashmatch

#endif // COSMICTRACKSELECTOR_H