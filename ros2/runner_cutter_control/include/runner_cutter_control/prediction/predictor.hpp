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
   * @param timestampSec Timestamp (in seconds) associated with the measurement.
   * @param measurement Measurement taken at the timestamp, which consists
   * of (x, y, z) position and confidence score.
   */
  virtual void add(double timestampSec, const Measurement& measurement) {
    history_[timestampSec] = measurement;
    lastTimestampSec_ = timestampSec;
  }

  /**
   * Predict the position at the given timestamp.
   *
   * @param timestampSec Timestamp (in seconds) to predict the measurement for.
   */
  virtual Position predict(double timestampSec) const = 0;

  /**
   * Clear the predictor's state.
   */
  virtual void reset() {
    history_.clear();
    lastTimestampSec_ = 0.0;
  }

  const std::map<double, Measurement>& getHistory() const { return history_; }
  double getLastTimestampSec() const { return lastTimestampSec_; }

 protected:
  /**
   * Get linearly interpolated position at a given timestamp.
   * If timestamp is outside the historical range, returns the nearest stored
   * value.
   *
   * @param timestampSec Timestamp (in seconds) to get the measurement for.
   */
  Position interpolated(double timestampSec) const;

  std::map<double, Measurement> history_;
  double lastTimestampSec_{0.0};
};