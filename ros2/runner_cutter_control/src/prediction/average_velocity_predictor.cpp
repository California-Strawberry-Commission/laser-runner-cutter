#include "runner_cutter_control/prediction/average_velocity_predictor.hpp"

Position AverageVelocityPredictor::predict(double timestampSec) const {
  double dt{timestampSec - getLastTimestampSec()};

  if (dt <= 0.0) {
    return interpolated(timestampSec);
  }

  const auto& history{getHistory()};

  if (history.empty()) {
    return {0, 0, 0};
  }

  if (history.size() < 2) {
    return history.rbegin()->second.position;
  }

  const auto& [t0, m0]{*history.begin()};   // oldest
  const auto& [t1, m1]{*history.rbegin()};  // most recent

  if ((t1 - t0) <= 0.0) {
    return m1.position;
  }

  return {m1.position.x + static_cast<float>((m1.position.x - m0.position.x) /
                                             (t1 - t0) * (timestampSec - t1)),
          m1.position.y + static_cast<float>((m1.position.y - m0.position.y) /
                                             (t1 - t0) * (timestampSec - t1)),
          m1.position.z + static_cast<float>((m1.position.z - m0.position.z) /
                                             (t1 - t0) * (timestampSec - t1))};
}
