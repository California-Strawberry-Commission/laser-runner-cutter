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
   * @param timestampMs Timestamp (in ms) associated with the measurement.
   * @param measurement Measurement taken at the timestamp, which consists
   * of (x, y, z) position and confidence score.
   */
  void add(double timestampMs, const Measurement& measurement) override;

  /**
   * Predict the position at the given timestamp. If the timestamp provided
   * is earlier than that of the last measurement, interpolate based on the
   * historical measurements.
   *
   * @param timestampMs Timestamp (in ms) to predict the measurement for.
   */
  Position predict(double timestampMs) const override;

  /**
   * Clear the predictor's state.
   */
  void reset() override;

 private:
  Eigen::MatrixXd lerpR(float confidence, float min, float max);

  Eigen::MatrixXd F_, H_, P_, R_, Q_;
  Eigen::VectorXd x_;
  bool initialized_{false};
};