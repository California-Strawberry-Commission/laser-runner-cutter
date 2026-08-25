#pragma once

#include <cstddef>
#include <utility>

#include "common/ring_buffer.hpp"
#include "runner_cutter_control/common_types.hpp"

class Predictor {
 public:
  struct Measurement {
    Position position;
    float confidence;
  };

  using HistoryEntry = std::pair<double, Measurement>;

  explicit Predictor(std::size_t maxHistorySize = 1024)
      : history_{maxHistorySize} {}

  virtual ~Predictor() = default;

  /**
   * Add a new position measurement to the predictor. Measurements with a
   * timestamp at or before the last added measurement's timestamp are
   * considered out-of-order and are ignored. Once the history is at capacity,
   * adding a measurement evicts the oldest one.
   *
   * @param timestampSec Timestamp, in seconds, associated with the measurement.
   * @param position Position (x, y, z) measurement.
   * @param confidence Confidence score associated with the position
   * measurement.
   * @return True if the measurement was added, false if ignored (e.g. out of
   * order).
   */
  virtual bool add(double timestampSec, const Position& position,
                   float confidence) {
    if (!history_.empty() && timestampSec <= lastTimestampSec_) {
      return false;
    }

    history_.push_back({timestampSec, {position, confidence}});
    lastTimestampSec_ = timestampSec;
    return true;
  }

  /**
   * Add a velocity-only measurement update.
   *
   * @param timestampSec Timestamp, in seconds, associated with the
   * measurement.
   * @param velocity Velocity (vx, vy, vz) measurement.
   * @param confidence Confidence score associated with the velocity
   * measurement.
   * @return True if the measurement was applied, false if unsupported or
   * ignored (e.g. out of order).
   */
  virtual bool addVelocity(double /*timestampSec*/,
                           const Position& /*velocity*/, float /*confidence*/) {
    return false;
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
