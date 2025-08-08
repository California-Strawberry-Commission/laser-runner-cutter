#pragma once

#include <Eigen/Dense>

#include "runner_cutter_control/prediction/predictor.hpp"

/**
 * Predictor that uses a Kalman Filter in order to predict future
 * measurements.
 */
class KalmanFilterPredictor final : public Predictor {
 public:
  KalmanFilterPredictor();
  ~KalmanFilterPredictor() = default;

  /**
   * Add a new position measurement to the predictor. Calls to `add` for
   * measurements must be done in sequential order with respect to their
   * timestamps.
   *
   * @param timestampSec Timestamp (in seconds) associated with the measurement.
   * @param measurement Measurement taken at the timestamp, which consists
   * of (x, y, z) position and confidence score.
   */
  void add(double timestampSec, const Measurement& measurement) override;

  /**
   * Predict the position at the given timestamp. If the timestamp provided
   * is earlier than that of the last measurement, interpolate based on the
   * historical measurements.
   *
   * @param timestampSec Timestamp (in seconds) to predict the measurement for.
   */
  Position predict(double timestampSec) const override;

  /**
   * Clear the predictor's state.
   */
  void reset() override;

 private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Matrix3x6d = Eigen::Matrix<double, 3, 6>;

  void predictStep(double dt);
  void updateStep(const Eigen::Vector3d& z, float confidence);
  Eigen::Matrix3d lerpR(float confidence, double min, double max);

  // State vector (x): x, y, z, vx, vy, vz
  Vector6d x_;

  // State transition matrix (F)
  Matrix6d F_;

  // Measurement function (H) - extracts measurement from state
  Matrix3x6d H_;

  // State covariance matrix (P): confidence in the estimate
  Matrix6d P_;

  // Process noise covariance (Q): how much we expect motion to vary over time
  // Lower = smoother, more stable, but slower response to movement changes
  // Higher = faster response to movement changes
  Matrix6d Q_;

  // Measurement noise covariance (R): uncertainty in measurements
  // Lower = trust provided measurements more
  // Higher = trust model more
  Eigen::Matrix3d R_;

  bool initialized_{false};
};