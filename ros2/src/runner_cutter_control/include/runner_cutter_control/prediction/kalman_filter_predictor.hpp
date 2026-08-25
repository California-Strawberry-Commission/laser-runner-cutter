#pragma once

#include <Eigen/Dense>

#include "runner_cutter_control/prediction/predictor.hpp"

/**
 * Predictor that uses a Kalman Filter in order to predict future
 * measurements.
 */
class KalmanFilterPredictor final : public Predictor {
 public:
  /**
   * @param positionMeasurementNoiseStdMin Position measurement noise std dev at
   * max (1.0) confidence
   * @param positionMeasurementNoiseStdMax Position measurement noise std dev at
   * min (0.0) confidence
   * @param accelerationNoiseStd Acceleration noise std dev
   * @param initialPositionStd Initial position uncertainty
   * @param initialVelocityStd Initial velocity uncertainty
   * @param velocityMeasurementNoiseStdMin Velocity measurement noise std dev
   * at max (1.0) confidence
   * @param velocityMeasurementNoiseStdMax Velocity measurement noise std dev
   * at min (0.0) confidence
   */
  KalmanFilterPredictor(double positionMeasurementNoiseStdMin = 4.0,
                        double positionMeasurementNoiseStdMax = 20.0,
                        double accelerationNoiseStd = 300.0,
                        double initialPositionStd = 1000.0,
                        double initialVelocityStd = 500.0,
                        double velocityMeasurementNoiseStdMin = 50.0,
                        double velocityMeasurementNoiseStdMax = 250.0);
  ~KalmanFilterPredictor() = default;

  /**
   * Add a new position measurement to the predictor. Measurements with a
   * timestamp at or before the last added measurement's timestamp are
   * considered out-of-order and are ignored.
   *
   * @param timestampSec Timestamp, in seconds, associated with the measurement.
   * @param position Position (x, y, z) measurement.
   * @param confidence Confidence score associated with the position
   * measurement.
   * @return True if the measurement was added, and false if it was ignored
   * for being out of order.
   */
  bool add(double timestampSec, const Position& position,
           float confidence) override;

  /**
   * Add a velocity-only measurement update (no position measurement
   * available), e.g. derived from optical flow displacement when a track
   * was not detected this frame. Runs a predict step followed by a
   * correction step that only observes the velocity components of the
   * state; the position estimate still benefits via the position-velocity
   * cross-covariance.
   *
   * @param timestampSec Timestamp, in seconds, associated with the
   * measurement.
   * @param velocity Velocity (vx, vy, vz) measurement.
   * @param confidence Confidence score associated with the velocity
   * measurement.
   * @return True if the measurement was applied, false if the predictor
   * hasn't been initialized with a position yet, or if the timestamp is out
   * of order.
   */
  bool addVelocity(double timestampSec, const Position& velocity,
                   float confidence) override;

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
  void correctStep(const Eigen::Vector3d& z, const Matrix3x6d& H,
                   double noiseStdMin, double noiseStdMax, float confidence);

  double measurementNoiseStdMin_;
  double measurementNoiseStdMax_;
  double accelerationNoiseStd_;
  double initialPositionStd_;
  double initialVelocityStd_;
  double velocityMeasurementNoiseStdMin_;
  double velocityMeasurementNoiseStdMax_;

  // State vector (x): x, y, z, vx, vy, vz
  Vector6d x_;

  // State transition matrix (F)
  Matrix6d F_;

  // Measurement function (H): extracts position from state
  Matrix3x6d H_;

  // Measurement function for velocity-only updates (Hvel): extracts velocity
  // from state
  Matrix3x6d Hvel_;

  // State covariance matrix (P): confidence in the estimate
  Matrix6d P_;

  // Process noise covariance (Q): how much we expect motion to vary over time
  // Lower = smoother, more stable, but slower response to movement changes
  // Higher = faster response to movement changes
  Matrix6d Q_;

  bool initialized_{false};
};