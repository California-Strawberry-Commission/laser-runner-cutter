#include <iostream>

#include "matplotlibcpp.h"
#include "runner_cutter_control/prediction/average_velocity_predictor.hpp"
#include "runner_cutter_control/prediction/kalman_filter_predictor.hpp"

namespace plt = matplotlibcpp;

int main() {
  // AverageVelocityPredictor predictor;
  KalmanFilterPredictor predictor;

  // Moving
  std::vector<std::pair<double, Position>> data{
      {1740157598942.124, {41.0, -64.5, 432.75}},
      {1740157599001.046, {42.25, -70.75, 446.25}},
      {1740157599059.6873, {42.0, -74.75, 434.0}},
      {1740157599118.7654, {43.25, -81.5, 447.5}},
      {1740157599177.7407, {43.0, -85.25, 444.5}},
      {1740157599236.8506, {42.75, -88.5, 442.75}},
      {1740157599297.516, {42.5, -94.25, 437.25}},
      {1740157599356.225, {40.5, -94.5, 426.75}},
      {1740157599414.5542, {40.5, -97.75, 426.5}},
  };

  /*
  // Static
  std::vector<std::pair<double, Position>> data{
      {1740155698317.263, {51.75, -75.5, 427.0}},
      {1740155698376.7227, {52.25, -77.5, 431.5}},
      {1740155698434.172, {51.5, -75.5, 426.5}},
      {1740155698493.7585, {51.75, -75.75, 428.5}},
      {1740155698551.7332, {52.0, -76.5, 430.25}},
      {1740155698610.8508, {51.5, -75.5, 426.25}},
      {1740155698671.3154, {51.75, -76.5, 427.0}},
      {1740155698729.7744, {51.75, -76.5, 428.0}},
      {1740155698789.1252, {52.0, -76.5, 430.25}},
  };
  */

  std::vector<float> measX, measY, predX, predY;

  for (const auto& [timestampMs, position] : data) {
    predictor.add(timestampMs / 1000.0, {position, 1.0f});
    measX.push_back(position.x);
    measY.push_back(position.y);
  }

  std::vector<double> timeDeltas;
  for (size_t i = 1; i < data.size(); ++i) {
    timeDeltas.push_back((data[i].first - data[i - 1].first) / 1000.0);
  }

  double avgTimeDelta{accumulate(timeDeltas.begin(), timeDeltas.end(), 0.0) /
                      timeDeltas.size()};
  double lastTimestampS{data.back().first / 1000.0};

  for (int i = 1; i <= 5; ++i) {
    double futureTimestampS{lastTimestampS + avgTimeDelta * i};
    auto [x, y, z]{predictor.predict(futureTimestampS)};
    predX.push_back(x);
    predY.push_back(y);
  }

  plt::figure_size(800, 600);
  plt::named_plot("Measurements", measX, measY, "ro");
  plt::named_plot("Predictions", predX, predY, "bo");
  plt::xlabel("X");
  plt::ylabel("Y");
  plt::xlim(0, 100);
  plt::ylim(-150, -50);
  plt::legend();
  plt::title("Future Position Predictions");
  plt::grid(true);
  plt::show();

  return 0;
}