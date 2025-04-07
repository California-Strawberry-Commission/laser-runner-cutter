#pragma once

#include <Eigen/Dense>

#include "runner_cutter_control_cpp/prediction/predictor.hpp"

class KalmanFilterPredictor final : public Predictor {
 public:
  KalmanFilterPredictor();

  /**
   * Add a new position measurement to the predictor. Calls to `add` for
   * measurements must be done in sequential order with respect to their
   * timestamps.
   *
   * @param position Position measurement (x, y, z).
   * @param timestampMs Timestamp (in ms) associated with the measurement.
   * @param confidence Confidence score associated with the measurement.
   */
  void add(std::tuple<float, float, float> position, double timestampMs,
           float confidence = 1.0f) override;

  /**
   * Predict a future position.
   *
   * @param timestampMs Timestamp (in ms) to predict the measurement for.
   */
  std::tuple<float, float, float> predict(double timestampMs) override;

  /**
   * Clear the predictor's state.
   */
  void reset() override;

 private:
  Eigen::MatrixXd lerpR(float confidence, float min, float max);

  Eigen::MatrixXd F_, H_, P_, R_, Q_;
  Eigen::VectorXd x_;
  double lastTimestampMs_{0.0f};
  bool initialized_{false};
};