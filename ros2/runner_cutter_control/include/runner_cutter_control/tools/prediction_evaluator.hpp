#pragma once

#include "runner_cutter_control/common_types.hpp"
#include "runner_cutter_control/prediction/predictor.hpp"

namespace prediction_evaluator {

double evaluatePredictor(std::unique_ptr<Predictor> predictor,
                         const std::vector<double>& timestampsMs,
                         const std::vector<Position>& positions,
                         const std::vector<float>& confidences,
                         double predictionOffsetMs) {
  if (timestampsMs.empty() || timestampsMs.size() != positions.size() ||
      timestampsMs.size() != confidences.size()) {
    return -1.0;
  }

  predictor->reset();
  std::vector<double> errors;
  for (size_t i = 0; i < timestampsMs.size(); ++i) {
    double t{timestampsMs[i]};
    const Position& pos{positions[i]};
    float confidence{confidences[i]};

    predictor->add(t, {pos, confidence});

    double predictTime{t + predictionOffsetMs};

    // Find interval [t_i, t_{i+1}] that contains predictTime
    auto it{std::upper_bound(timestampsMs.begin(), timestampsMs.end(),
                             predictTime)};
    if (it == timestampsMs.begin() || it == timestampsMs.end()) {
      continue;
    }
    size_t idx2{static_cast<size_t>(std::distance(timestampsMs.begin(), it))};
    size_t idx1{idx2 - 1};
    double t1{timestampsMs[idx1]};
    double t2{timestampsMs[idx2]};
    const Position& p1{positions[idx1]};
    const Position& p2{positions[idx2]};

    double alpha{(predictTime - t1) / (t2 - t1)};
    Position groundTruth{static_cast<float>(p1.x + alpha * (p2.x - p1.x)),
                         static_cast<float>(p1.y + alpha * (p2.y - p1.y)),
                         static_cast<float>(p1.z + alpha * (p2.z - p1.z))};

    Position predicted{predictor->predict(predictTime)};
    double error{std::sqrt(
        (groundTruth.x - predicted.x) * (groundTruth.x - predicted.x) +
        (groundTruth.y - predicted.y) * (groundTruth.y - predicted.y) +
        (groundTruth.z - predicted.z) * (groundTruth.z - predicted.z))};
    errors.push_back(error);
  }

  if (errors.empty()) {
    return -1.0;
  }

  double sum{std::accumulate(errors.begin(), errors.end(), 0.0)};
  return sum / errors.size();
}

}  // namespace prediction_evaluator