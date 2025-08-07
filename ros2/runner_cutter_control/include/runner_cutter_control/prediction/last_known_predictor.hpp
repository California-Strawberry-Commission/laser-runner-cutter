#pragma once

#include "runner_cutter_control/prediction/predictor.hpp"

/**
 * Predictor that always uses the last known measurement as the prediction.
 */
class LastKnownPredictor final : public Predictor {
 public:
  LastKnownPredictor() = default;
  ~LastKnownPredictor() = default;

  /**
   * Predict the position at the given timestamp. If the timestamp provided
   * is earlier than that of the last measurement, interpolate based on the
   * historical measurements.
   *
   * @param timestampMs Timestamp (in ms) to predict the measurement for.
   */
  Position predict(double timestampMs) const override;
};