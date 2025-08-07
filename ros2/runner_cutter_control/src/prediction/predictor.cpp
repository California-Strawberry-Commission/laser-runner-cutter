#include "runner_cutter_control/prediction/predictor.hpp"

Position Predictor::interpolated(double timestampMs) const {
  if (history_.empty()) {
    return {0.0f, 0.0f, 0.0f};
  }

  auto it{history_.lower_bound(timestampMs)};
  if (it == history_.begin()) {
    const auto& pos{it->second.position};
    return {pos.x, pos.y, pos.z};
  } else if (it == history_.end()) {
    const auto& pos{std::prev(it)->second.position};
    return {pos.x, pos.y, pos.z};
  } else {
    auto it2{it};
    auto it1{std::prev(it)};
    double t1{it1->first};
    double t2{it2->first};
    const auto& p1{it1->second.position};
    const auto& p2{it2->second.position};

    double alpha{(timestampMs - t1) / (t2 - t1)};
    float x{static_cast<float>((1.0 - alpha) * p1.x + alpha * p2.x)};
    float y{static_cast<float>((1.0 - alpha) * p1.y + alpha * p2.y)};
    float z{static_cast<float>((1.0 - alpha) * p1.z + alpha * p2.z)};
    return {x, y, z};
  }
}
