#pragma once

#include <Eigen/Dense>
#include <optional>

#include "runner_cutter_control/common_types.hpp"

/**
 * Geometric model of a dual-mirror galvo laser scanner, mapping 3D positions in
 * camera-space to laser coords.
 *
 * Unlike a pinhole projector, a galvo scanner:
 * - Commands mirror angles (laser coords are linear in angle, not tan(angle)).
 * - Deflects the beam with two mirrors separated by some distance, so there is
 *   no single center of projection and the first mirror's deflection is
 *   stretched by the second (pincushion distortion).
 *
 * The scanner frame (the laser projector's 3D coordinate system) has its origin
 * on the rotation axis of the second (exit) mirror, z pointing forward. The
 * first mirror sits `mirrorDistance` behind the second one along the beam path.
 * For a point (a, b, z) in the scanner frame, where `a` is along the first
 * mirror's deflection direction and `b` along the second (exit) mirror's,
 *   exitMirrorAngle = atan2(b, z)
 *   firstMirrorAngle = atan2(a, sqrt(b^2 + z^2) + mirrorDistance)
 * and each laser coord axis is an affine function of its mirror's angle.
 *
 * Model parameters:
 * - rotation (3): camera-to-scanner rotation R as an angle-axis vector (rad),
 * such that p_scanner = R * p_camera + t
 * - translation (3): camera-to-scanner translation t (mm), such that
 *   p_scanner = R * p_camera + t.
 * - scale (2): laser coord units per radian of optical angle, for laser x and
 *   y. Negative if the laser axis is inverted relative to the scanner frame.
 * - offset (2): laser coord at zero mirror angle, for laser x and y.
 * - mirrorDistance (1): distance from the first mirror to the exit mirror (mm).
 */
struct GalvoModel {
  static constexpr int NUM_PARAMS{11};

  struct Ray {
    Eigen::Vector3d origin;
    // Unit vector
    Eigen::Vector3d direction;
  };

  // Scanner pose relative to the camera. A camera-space position maps to the
  // scanner frame as p_scanner = R * p_camera + t.
  Eigen::Vector3d rotation{Eigen::Vector3d::Zero()};
  Eigen::Vector3d translation{Eigen::Vector3d::Zero()};

  // Mapping from mirror angle to laser coord, per laser axis (x, y):
  //   laser coord = scale * angle + offset
  // where angle is the optical deflection (twice the mechanical mirror angle),
  // in radians.
  Eigen::Vector2d scale{Eigen::Vector2d::Zero()};
  Eigen::Vector2d offset{Eigen::Vector2d::Zero()};

  // Distance along the beam from the first mirror to the exit
  // mirror, in camera-space units (mm).
  double mirrorDistance{0.0};

  // Whether laser x drives the first mirror the beam hits, with laser y driving
  // the exit mirror. For a standard ILDA scanner, the first mirror deflects
  // left/right and the exit mirror deflects up/down.
  bool xMirrorFirst{true};

  /**
   * Transform a 3D position in camera-space to a laser coord.
   *
   * @return (x, y) laser coord, or nullopt if the position is not in front of
   * the scanner.
   */
  std::optional<Eigen::Vector2d> project(const Eigen::Vector3d& position) const;
  std::optional<LaserCoord> project(const Position& position) const;

  /**
   * The beam emitted for a laser coord, in camera-space.
   */
  Ray backproject(const Eigen::Vector2d& laserCoord) const;

  /**
   * Distance (in camera-space units) from a position to the beam emitted for
   * a laser coord, i.e. how far the laser would miss the position.
   */
  double distanceToBeam(const Eigen::Vector2d& laserCoord,
                        const Eigen::Vector3d& position) const;

  /**
   * Model parameters, flattened as [rotation (3), translation (3), scale (2),
   * offset (2), mirrorDistance (1)].
   */
  Eigen::VectorXd getParams() const;
  void setParams(const Eigen::VectorXd& params);
};
