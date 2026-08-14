#pragma once

#include <cstddef>
#include <utility>

#include "ring_buffer.hpp"
#include "runner_cutter_control/common_types.hpp"

class Predictor {
 public:
  struct Measurement {
    Position position;
    float confidence;
  };

  using HistoryEntry = std::pair<double, Measurement>;
  static constexpr std::size_t MAX_PREDICTOR_POINTS{1024};

  explicit Predictor(std::size_t maxHistorySize = MAX_PREDICTOR_POINTS)
      : history_{maxHistorySize} {}

  virtual ~Predictor() = default;

  /**
   * Add a new position measurement to the predictor. Measurements with a
   * timestamp at or before the last added measurement's timestamp are
   * considered out-of-order and are ignored. Once the history is at capacity,
   * adding a measurement evicts the oldest one.
   *
   * @param timestampSec Timestamp, in seconds, associated with the measurement.
   * @param measurement Measurement taken at the timestamp, which consists
   * of (x, y, z) position and confidence score.
   * @return True if the measurement was added, and false if it was ignored
   * for being out of order.
   */
  virtual bool add(double timestampSec, const Measurement& measurement) {
    if (!history_.empty() && timestampSec <= lastTimestampSec_) {
      return false;
    }

    history_.push_back({timestampSec, measurement});
    lastTimestampSec_ = timestampSec;
    return true;
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

  const RingBuffer<HistoryEntry>& getHistory() const { return history_; }
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

  RingBuffer<HistoryEntry> history_;
  double lastTimestampSec_{0.0};
};
