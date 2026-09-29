#pragma once

#include <Eigen/Dense>
#include <optional>
#include <tuple>
#include <vector>

#include "runner_cutter_control/calibration/galvo_model.hpp"
#include "runner_cutter_control/common_types.hpp"

class PointCorrespondences {
 public:
  // The galvo model has 11 parameters, and each correspondence provides 2
  // constraints
  static constexpr std::size_t MIN_CORRESPONDENCES{6};

  struct FitStats {
    std::size_t numInliers{0};
    // Laser coord error between the observed and projected laser coords
    float meanLaserError{0.0f};
    float maxLaserError{0.0f};
    // Distance (camera-space units, i.e. mm) between each observed position
    // and the beam the model predicts for the observed laser coord
    float meanPositionError{0.0f};
    float maxPositionError{0.0f};
    // Range of camera-space z among the inliers
    float minDepth{0.0f};
    float maxDepth{0.0f};
    // Whether the positions lie close to a single plane, in which case the
    // model is poorly constrained away from that plane
    bool nearlyPlanar{false};
  };

  PointCorrespondences() = default;
  ~PointCorrespondences() = default;

  std::size_t size() const;

  /**
   * Add a point correspondence: a laser coord that was shot, and where the
   * laser spot was observed by the camera.
   *
   * Note: the laser bounds are updated immediately, but updateModel() must be
   * called manually after all point correspondences have been added.
   *
   * @param laserCoord Laser coord (x, y) that was commanded, in the DAC's
   * normalized [0, 1] range.
   * @param cameraPixelCoord Pixel (u, v) of the detected laser spot in the
   * color image. Used for the laser bounds and the pixel-to-laser Jacobian.
   * @param cameraPosition 3D position (x, y, z) of the detected laser spot in
   * camera-space (mm). Used to fit the model.
   */
  void add(const LaserCoord& laserCoord, const PixelCoord& cameraPixelCoord,
           const Position& cameraPosition);
  void clear();

  /**
   * Update everything derived from the point correspondences: the laser
   * bounds, the camera pixel to laser coord Jacobian, and the galvo model.
   */
  void updateModel();

  /**
   * @return Whether a model has been fit successfully.
   */
  bool hasModel() const { return hasModel_; }

  /**
   * Transform a 3D position in camera-space to a laser coord using the fitted
   * model.
   *
   * @return (x, y) laser coord, or nullopt if there is no model or the
   * position is not in front of the scanner.
   */
  std::optional<LaserCoord> project(const Position& cameraPosition) const;

  const GalvoModel& getModel() const { return model_; }
  FitStats getFitStats() const { return fitStats_; }

  /**
   * @return Mean laser coord error across inlier correspondences.
   */
  float getReprojectionError() const { return fitStats_.meanLaserError; }

  /**
   * The rect (min x, min y, width, height) representing the reach of the laser,
   * in terms of camera pixels.
   *
   * @return Tuple representing (min x, min y, width, height) of the laser
   * bounds.
   */
  PixelRect getLaserBounds() const { return laserBounds_; }

  /**
   * Get the Jacobian from camera pixels to laser coords.
   *
   * @return Jacobian matrix.
   */
  Eigen::Matrix2d getCameraPixelToLaserCoordJacobian() const {
    return cameraToLaserJacobian_;
  }

  /**
   * Write the point correspondences to a stream.
   */
  void serialize(std::ostream& os) const;

  /**
   * Replace the point correspondences with those read from a stream, and fit
   * the model.
   */
  void deserialize(std::istream& is);

 private:
  std::vector<LaserCoord> laserCoords_;
  std::vector<PixelCoord> cameraPixelCoords_;
  std::vector<Position> cameraPositions_;
  GalvoModel model_{};
  bool hasModel_{false};
  FitStats fitStats_{};
  PixelRect laserBounds_{0, 0, 0, 0};
  Eigen::Matrix2d cameraToLaserJacobian_{Eigen::Matrix2d::Zero()};

  void updateLaserBounds();
  void updateCameraPixelToLaserCoordJacobian();
  void fitGalvoModel();
};
