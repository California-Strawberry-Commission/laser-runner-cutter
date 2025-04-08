#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <vector>

class PointCorrespondences {
 public:
  PointCorrespondences();
  std::size_t size() const;
  void add(std::pair<float, float> laserCoord,
           std::pair<int, int> cameraPixelCoord,
           std::tuple<float, float, float> cameraPosition);
  void clear();
  void updateTransformLinearLeastSquares();
  void updateTransformNonlinearLeastSquares();
  float getReprojectionError() const;

  /**
   * The transform matrix that converts a 3D position in camera-space to laser
   * coordinates.
   *
   * @return Transform matrix that converts a 3D position in camera-space to
   * laser coordinates.
   */
  Eigen::MatrixXd getCameraToLaserTransform() const;

  /**
   * The rect (min x, min y, width, height) representing the reach of the laser,
   * in terms of camera pixels.
   *
   * @return Tuple representing (min x, min y, width, height) of the laser
   * bounds.
   */
  std::tuple<int, int, int, int> getLaserBounds() const;

  void serialize(std::ostream& os) const;
  void deserialize(std::istream& is);

 private:
  std::vector<std::pair<float, float>> laserCoords_;
  std::vector<std::pair<int, int>> cameraPixelCoords_;
  std::vector<std::tuple<float, float, float>> cameraPositions_;
  Eigen::MatrixXd cameraToLaserTransform_;
  std::tuple<int, int, int, int> laserBounds_{0, 0, 0, 0};

  void updateLaserBounds();
};
