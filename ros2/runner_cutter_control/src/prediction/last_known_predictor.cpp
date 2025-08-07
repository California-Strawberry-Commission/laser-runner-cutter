#include "runner_cutter_control/prediction/last_known_predictor.hpp"

Position LastKnownPredictor::predict(double timestampMs) const {
  double dt{timestampMs - getLastTimestampMs()};

  if (dt <= 0.0) {
    return interpolated(timestampMs);
  }

  const auto& history{getHistory()};

  if (history.empty()) {
    return {0, 0, 0};
  }

  return history.rbegin()->second.position;
}
