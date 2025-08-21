#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <vector>

#include "runner_cutter_control/common_types.hpp"

class PointCorrespondences {
 public:
  PointCorrespondences();
  ~PointCorrespondences() = default;

  std::size_t size() const;
  void add(const LaserCoord& laserCoord, const PixelCoord& cameraPixelCoord,
           const Position& cameraPosition);
  void clear();
  void updateTransformLinearLeastSquares();
  void updateTransformNonlinearLeastSquares();
  float getReprojectionError() const;

  /**
   * Update the Jacobian from camera pixels to laser coords using all point
   * correspondences.
   */
  void updateCameraPixelToLaserCoordJacobian();

  /**
   * The transform matrix that converts a 3D position in camera-space to laser
   * coordinates.
   *
   * @return Transform matrix that converts a 3D position in camera-space to
   * laser coordinates.
   */
  Eigen::MatrixXd getCameraToLaserTransform() const {
    return cameraToLaserTransform_;
  }

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

  void serialize(std::ostream& os) const;
  void deserialize(std::istream& is);

 private:
  std::vector<LaserCoord> laserCoords_;
  std::vector<PixelCoord> cameraPixelCoords_;
  std::vector<Position> cameraPositions_;
  Eigen::MatrixXd cameraToLaserTransform_;
  PixelRect laserBounds_{0, 0, 0, 0};
  Eigen::Matrix2d cameraToLaserJacobian_;

  void updateLaserBounds();
};
