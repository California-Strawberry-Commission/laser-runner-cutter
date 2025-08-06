#pragma once

#include <Eigen/Dense>
#include <map>

#include "runner_cutter_control/prediction/predictor.hpp"

class KalmanFilterPredictor final : public Predictor {
 public:
  KalmanFilterPredictor();

  /**
   * Add a new position measurement to the predictor. Calls to `add` for
   * measurements must be done in sequential order with respect to their
   * timestamps. Non-sequential measurements will be ignored.
   *
   * @param position Position measurement (x, y, z).
   * @param timestampMs Timestamp (in ms) associated with the measurement.
   * @param confidence Confidence score associated with the measurement.
   */
  void add(const Position& position, double timestampMs,
           float confidence = 1.0f) override;

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

  /**
   * Get linearly interpolated position at a given timestamp.
   * If timestamp is outside the historical range, returns the nearest stored
   * value.
   *
   * @param timestampMs Timestamp (in ms) to get the measurement for.
   */
  Position interpolated(double timestampMs) const;

  Eigen::MatrixXd F_, H_, P_, R_, Q_;
  Eigen::VectorXd x_;
  double lastTimestampMs_{0.0f};
  bool initialized_{false};
  std::map<double, Position> history_;
};