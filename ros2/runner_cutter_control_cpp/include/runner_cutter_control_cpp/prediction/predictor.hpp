#pragma once

#include <tuple>

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
  virtual void add(std::tuple<float, float, float> position, double timestampMs,
                   float confidence = 1.0f) = 0;

  /**
   * Predict a future position.
   *
   * @param timestampMs Timestamp (in ms) to predict the measurement for.
   */
  virtual std::tuple<float, float, float> predict(double timestampMs) = 0;

  /**
   * Clear the predictor's state.
   */
  virtual void reset() = 0;
};