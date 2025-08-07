#pragma once

#include <map>

#include "runner_cutter_control/common_types.hpp"

class Predictor {
 public:
  struct Measurement {
    Position position;
    float confidence;
  };

  /**
   * Add a new position measurement to the predictor. Calls to `add` for
   * measurements must be done in sequential order with respect to their
   * timestamps.
   *
   * @param timestampMs Timestamp (in ms) associated with the measurement.
   * @param measurement Measurement taken at the timestamp, which consists
   * of (x, y, z) position and confidence score.
   */
  virtual void add(double timestampMs, const Measurement& measurement) {
    history_[timestampMs] = measurement;
    lastTimestampMs_ = timestampMs;
  }

  /**
   * Predict the position at the given timestamp.
   *
   * @param timestampMs Timestamp (in ms) to predict the measurement for.
   */
  virtual Position predict(double timestampMs) const = 0;

  /**
   * Clear the predictor's state.
   */
  virtual void reset() {
    history_.clear();
    lastTimestampMs_ = 0.0;
  }

  const std::map<double, Measurement>& getHistory() const { return history_; }
  double getLastTimestampMs() const { return lastTimestampMs_; }

 protected:
  /**
   * Get linearly interpolated position at a given timestamp.
   * If timestamp is outside the historical range, returns the nearest stored
   * value.
   *
   * @param timestampMs Timestamp (in ms) to get the measurement for.
   */
  Position interpolated(double timestampMs) const;

  std::map<double, Measurement> history_;
  double lastTimestampMs_{0.0};
};