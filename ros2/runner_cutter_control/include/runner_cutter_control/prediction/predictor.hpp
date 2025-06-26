#pragma once

#include <tuple>

#include "runner_cutter_control/common_types.hpp"

class Predictor {
 public:
  /**
   * Add a new position measurement to the predictor. Calls to `add` for
   * measurements must be done in sequential order with respect to their
   * timestamps.
   *
   * @param position Position measurement (x, y, z).
   * @param timestampMs Timestamp (in ms) associated with the measurement.
   * @param confidence Confidence score associated with the measurement.
   */
  virtual void add(const Position& position, double timestampMs,
                   float confidence = 1.0f) = 0;

  /**
   * Predict a future position.
   *
   * @param timestampMs Timestamp (in ms) to predict the measurement for.
   */
  virtual Position predict(double timestampMs) = 0;

  /**
   * Clear the predictor's state.
   */
  virtual void reset() = 0;
};