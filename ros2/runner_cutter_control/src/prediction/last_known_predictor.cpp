#include "runner_cutter_control/prediction/last_known_predictor.hpp"

Position LastKnownPredictor::predict(double timestampSec) const {
  double dt{timestampSec - getLastTimestampSec()};

  if (dt <= 0.0) {
    return interpolated(timestampSec);
  }

  const auto& history{getHistory()};

  if (history.empty()) {
    return {0, 0, 0};
  }

  return history.rbegin()->second.position;
}
