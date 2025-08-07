#pragma once

#include "runner_cutter_control/prediction/predictor.hpp"

/**
 * Predictor that calculates the average velocity based on historical
 * measurements in order to predict future measurements.
 */
class AverageVelocityPredictor final : public Predictor {
 public:
  AverageVelocityPredictor() = default;
  ~AverageVelocityPredictor() = default;

  /**
   * Predict the position at the given timestamp. If the timestamp provided
   * is earlier than that of the last measurement, interpolate based on the
   * historical measurements.
   *
   * @param timestampMs Timestamp (in ms) to predict the measurement for.
   */
  Position predict(double timestampMs) const override;
};