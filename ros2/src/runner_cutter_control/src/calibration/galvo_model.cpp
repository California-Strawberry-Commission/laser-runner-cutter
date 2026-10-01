#include "runner_cutter_control/calibration/galvo_model.hpp"

#include <cmath>

namespace {

// Converts between scanner-frame (x, y, z) and the canonical (a, b, z) frame,
// where a is along the first mirror's deflection direction and b along the
// second mirror's. The mapping is its own inverse.
Eigen::Vector3d swapToCanonical(const Eigen::Vector3d& v, bool xMirrorFirst) {
  return xMirrorFirst ? v : Eigen::Vector3d{v.y(), v.x(), v.z()};
}

Eigen::Matrix3d rotationMatrix(const Eigen::Vector3d& angleAxis) {
  double angle{angleAxis.norm()};
  if (angle < 1e-12) {
    return Eigen::Matrix3d::Identity();
  }
  return Eigen::AngleAxisd{angle, angleAxis / angle}.toRotationMatrix();
}

}  // namespace

std::optional<Eigen::Vector2d> GalvoModel::project(
    const Eigen::Vector3d& position) const {
  // p_scanner = R * p_camera + t
  Eigen::Vector3d scannerPosition{rotationMatrix(rotation) * position +
                                  translation};
  Eigen::Vector3d canonical{swapToCanonical(scannerPosition, xMirrorFirst)};
  double a{canonical.x()};
  double b{canonical.y()};
  double z{canonical.z()};
  if (z <= 0.0) {
    return std::nullopt;
  }

  double exitAngle{std::atan2(b, z)};
  double firstAngle{std::atan2(a, std::hypot(b, z) + mirrorDistance)};

  int firstAxis{xMirrorFirst ? 0 : 1};
  int exitAxis{1 - firstAxis};
  // laser coord = scale * angle + offset
  Eigen::Vector2d laserCoord;
  laserCoord[firstAxis] = scale[firstAxis] * firstAngle + offset[firstAxis];
  laserCoord[exitAxis] = scale[exitAxis] * exitAngle + offset[exitAxis];
  return laserCoord;
}

std::optional<LaserCoord> GalvoModel::project(const Position& position) const {
  auto laserCoordOpt{
      project(Eigen::Vector3d{position.x, position.y, position.z})};
  if (!laserCoordOpt) {
    return std::nullopt;
  }
  return LaserCoord{static_cast<float>(laserCoordOpt->x()),
                    static_cast<float>(laserCoordOpt->y())};
}

GalvoModel::Ray GalvoModel::backproject(
    const Eigen::Vector2d& laserCoord) const {
  int firstAxis{xMirrorFirst ? 0 : 1};
  int exitAxis{1 - firstAxis};
  double firstAngle{(laserCoord[firstAxis] - offset[firstAxis]) /
                    scale[firstAxis]};
  double exitAngle{(laserCoord[exitAxis] - offset[exitAxis]) / scale[exitAxis]};

  // The first mirror deflects the beam within the a-z plane, onto the exit
  // mirror's rotation axis (the a axis). The exit mirror then rotates the beam
  // about that axis.
  Eigen::Vector3d canonicalOrigin{mirrorDistance * std::tan(firstAngle), 0.0,
                                  0.0};
  Eigen::Vector3d canonicalDirection{
      std::sin(firstAngle), std::cos(firstAngle) * std::sin(exitAngle),
      std::cos(firstAngle) * std::cos(exitAngle)};

  // Scanner frame to camera frame
  Eigen::Matrix3d cameraToScanner{rotationMatrix(rotation)};
  Eigen::Vector3d origin{
      cameraToScanner.transpose() *
      (swapToCanonical(canonicalOrigin, xMirrorFirst) - translation)};
  Eigen::Vector3d direction{cameraToScanner.transpose() *
                            swapToCanonical(canonicalDirection, xMirrorFirst)};
  return {origin, direction};
}

double GalvoModel::distanceToBeam(const Eigen::Vector2d& laserCoord,
                                  const Eigen::Vector3d& position) const {
  Ray ray{backproject(laserCoord)};
  Eigen::Vector3d toPosition{position - ray.origin};
  if (toPosition.dot(ray.direction) < 0.0) {
    // Position is behind the scanner
    return toPosition.norm();
  }
  return toPosition.cross(ray.direction).norm();
}

Eigen::VectorXd GalvoModel::getParams() const {
  Eigen::VectorXd params{NUM_PARAMS};
  params << rotation, translation, scale, offset, mirrorDistance;
  return params;
}

void GalvoModel::setParams(const Eigen::VectorXd& params) {
  rotation = params.segment<3>(0);
  translation = params.segment<3>(3);
  scale = params.segment<2>(6);
  offset = params.segment<2>(8);
  mirrorDistance = params[10];
}
