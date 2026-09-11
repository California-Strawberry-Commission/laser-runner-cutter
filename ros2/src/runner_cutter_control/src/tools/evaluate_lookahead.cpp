// evaluate_lookahead
//
// Offline evaluation of lookahead prediction against a rosbag of
// DetectionResult messages.
//
//   1. Read the rosbag of DetectionResult messages.
//   2. Extract the ground-truth position time series for each track.
//   3. Replay every DetectionResult through DetectionTrackerUpdater and, each
//      frame, ask every track's predictor where it will be X ms in the future.
//   4. For each ground truth detection, compare it against that track's
//      lookahead prediction interpolated to the detection's own timestamp
//      (linear interpolation between nearby predictions) to calculate the
//      error.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "CLI/CLI.hpp"
#include "detection_interfaces/msg/detection_result.hpp"
#include "detection_interfaces/msg/detection_type.hpp"
#include "matplotlibcpp.h"
#include "opencv2/opencv.hpp"
#include "rclcpp/serialization.hpp"
#include "rclcpp/serialized_message.hpp"
#include "rclcpp/time.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "runner_cutter_control/common_types.hpp"
#include "runner_cutter_control/tasks/detection_tracker_updater.hpp"
#include "runner_cutter_control/tracking/track.hpp"
#include "runner_cutter_control/tracking/tracker.hpp"

namespace {

namespace di = detection_interfaces::msg;
namespace plt = matplotlibcpp;

struct Options {
  std::string bag;
  std::string topic{"/detection0/detections"};
  double trackMissTimeoutSecs{0.2};
  int targetAttempts{3};
  double lookaheadSecs{0.2};
  bool plot{false};
  double videoFps{10.0};
};

std::vector<di::DetectionResult> readRunnerDetections(
    const std::string& bag, const std::string& topic) {
  rosbag2_cpp::Reader reader;
  reader.open(bag);

  rclcpp::Serialization<di::DetectionResult> serialization;
  std::vector<di::DetectionResult> out;
  while (reader.has_next()) {
    const auto bagMsg{reader.read_next()};
    if (bagMsg->topic_name != topic) {
      continue;
    }
    rclcpp::SerializedMessage serialized{*bagMsg->serialized_data};
    di::DetectionResult msg;
    serialization.deserialize_message(&serialized, &msg);
    if (msg.detection_type == di::DetectionType::RUNNER) {
      out.push_back(std::move(msg));
    }
  }
  reader.close();

  std::stable_sort(
      out.begin(), out.end(),
      [](const di::DetectionResult& a, const di::DetectionResult& b) {
        return rclcpp::Time(a.timestamp) < rclcpp::Time(b.timestamp);
      });
  return out;
}

bool validPosition(const Position& p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
         !(p.x == 0.0f && p.y == 0.0f && p.z == 0.0f);
}

struct TrackSeries {
  std::vector<double> timestamps;  // monotonically increasing
  std::vector<Position> positions;
};

// Linear interpolation of `series` at `queryTimestamp`. Returns nullopt if
// `queryTimestamp` is outside the series' time span or the series has fewer
// than two samples.
std::optional<Position> interpolateAt(const TrackSeries& series,
                                      double queryTimestamp) {
  if (series.timestamps.size() < 2 ||
      queryTimestamp < series.timestamps.front() ||
      queryTimestamp > series.timestamps.back()) {
    return std::nullopt;
  }
  const auto hi{std::upper_bound(series.timestamps.begin(),
                                 series.timestamps.end(), queryTimestamp)};
  if (hi == series.timestamps.end()) {
    return series.positions.back();
  }
  const size_t i2{static_cast<size_t>(hi - series.timestamps.begin())};
  const size_t i1{i2 - 1};
  const double a{(queryTimestamp - series.timestamps[i1]) /
                 (series.timestamps[i2] - series.timestamps[i1])};
  const Position& p1{series.positions[i1]};
  const Position& p2{series.positions[i2]};
  return Position{static_cast<float>(p1.x + a * (p2.x - p1.x)),
                  static_cast<float>(p1.y + a * (p2.y - p1.y)),
                  static_cast<float>(p1.z + a * (p2.z - p1.z))};
}

double distance(const Position& a, const Position& b) {
  const double dx{static_cast<double>(a.x) - b.x};
  const double dy{static_cast<double>(a.y) - b.y};
  const double dz{static_cast<double>(a.z) - b.z};
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

struct EvaluatedPrediction {
  double timestamp{0.0};
  Position predicted{};
  Position actual{};
  double error{0.0};
};

constexpr int OUTPUT_VIDEO_WIDTH{900};
constexpr int OUTPUT_VIDEO_HEIGHT{900};
constexpr int OUTPUT_VIDEO_MARGIN_LEFT{90};
constexpr int OUTPUT_VIDEO_MARGIN_RIGHT{40};
constexpr int OUTPUT_VIDEO_MARGIN_TOP{60};
constexpr int OUTPUT_VIDEO_MARGIN_BOTTOM{70};

// Computes the X-Y bounds (with padding) spanning every ground truth and
// predicted position.
cv::Rect2d computeBounds(const std::map<uint32_t, TrackSeries>& groundTruths,
                         const std::map<uint32_t, TrackSeries>& predictions,
                         float padFactor = 0.1f) {
  double minX{std::numeric_limits<double>::infinity()};
  double maxX{-std::numeric_limits<double>::infinity()};
  double minY{std::numeric_limits<double>::infinity()};
  double maxY{-std::numeric_limits<double>::infinity()};
  for (const auto& kv : groundTruths) {
    for (const auto& p : kv.second.positions) {
      minX = std::min(minX, static_cast<double>(p.x));
      maxX = std::max(maxX, static_cast<double>(p.x));
      minY = std::min(minY, static_cast<double>(p.y));
      maxY = std::max(maxY, static_cast<double>(p.y));
    }
  }
  for (const auto& kv : predictions) {
    for (const auto& p : kv.second.positions) {
      minX = std::min(minX, static_cast<double>(p.x));
      maxX = std::max(maxX, static_cast<double>(p.x));
      minY = std::min(minY, static_cast<double>(p.y));
      maxY = std::max(maxY, static_cast<double>(p.y));
    }
  }

  if (minX > maxX || minY > maxY) {
    return cv::Rect2d{0.0, 0.0, 1.0, 1.0};
  }

  double spanX{maxX - minX};
  double spanY{maxY - minY};
  if (spanX < 1e-6) {
    spanX = 1.0;
  }
  if (spanY < 1e-6) {
    spanY = 1.0;
  }
  double padX{spanX * padFactor};
  double padY{spanY * padFactor};
  return cv::Rect2d{minX - padX, minY - padY, spanX + 2 * padX,
                    spanY + 2 * padY};
}

cv::Point toOutputVideoPixel(double x, double y, const cv::Rect2d& bounds) {
  constexpr int plotWidth{OUTPUT_VIDEO_WIDTH - OUTPUT_VIDEO_MARGIN_LEFT -
                          OUTPUT_VIDEO_MARGIN_RIGHT};
  constexpr int plotHeight{OUTPUT_VIDEO_HEIGHT - OUTPUT_VIDEO_MARGIN_TOP -
                           OUTPUT_VIDEO_MARGIN_BOTTOM};
  const double px{OUTPUT_VIDEO_MARGIN_LEFT +
                  (x - bounds.x) / bounds.width * plotWidth};
  // Flip Y so that larger values plot higher on the frame.
  const double py{OUTPUT_VIDEO_HEIGHT - OUTPUT_VIDEO_MARGIN_BOTTOM -
                  (y - bounds.y) / bounds.height * plotHeight};
  return cv::Point{static_cast<int>(std::lround(px)),
                   static_cast<int>(std::lround(py))};
}

cv::Mat renderOutputVideoFrame(
    double frameTimestamp, double t0,
    const std::vector<std::pair<uint32_t, Position>>& groundTruthPoints,
    const std::map<uint32_t, Position>& predictedPoints,
    const cv::Rect2d& bounds) {
  cv::Mat frame{OUTPUT_VIDEO_HEIGHT, OUTPUT_VIDEO_WIDTH, CV_8UC3,
                cv::Scalar(255, 255, 255)};

  const cv::Rect plotRect{
      OUTPUT_VIDEO_MARGIN_LEFT, OUTPUT_VIDEO_MARGIN_TOP,
      OUTPUT_VIDEO_WIDTH - OUTPUT_VIDEO_MARGIN_LEFT - OUTPUT_VIDEO_MARGIN_RIGHT,
      OUTPUT_VIDEO_HEIGHT - OUTPUT_VIDEO_MARGIN_TOP -
          OUTPUT_VIDEO_MARGIN_BOTTOM};
  cv::rectangle(frame, plotRect, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

  // Gridlines + tick labels
  const double minX{bounds.x};
  const double maxX{bounds.x + bounds.width};
  const double minY{bounds.y};
  const double maxY{bounds.y + bounds.height};
  constexpr int numDivisions{5};
  for (int i = 0; i <= numDivisions; ++i) {
    const double fx{minX + (maxX - minX) * i / numDivisions};
    const double fy{minY + (maxY - minY) * i / numDivisions};
    cv::line(frame, toOutputVideoPixel(fx, minY, bounds),
             toOutputVideoPixel(fx, maxY, bounds), cv::Scalar(220, 220, 220), 1,
             cv::LINE_AA);
    cv::line(frame, toOutputVideoPixel(minX, fy, bounds),
             toOutputVideoPixel(maxX, fy, bounds), cv::Scalar(220, 220, 220), 1,
             cv::LINE_AA);

    std::ostringstream xLabel;
    xLabel << std::fixed << std::setprecision(2) << fx;
    const cv::Point xTick{toOutputVideoPixel(fx, minY, bounds)};
    cv::putText(
        frame, xLabel.str(),
        {xTick.x - 18, OUTPUT_VIDEO_HEIGHT - OUTPUT_VIDEO_MARGIN_BOTTOM + 20},
        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    std::ostringstream yLabel;
    yLabel << std::fixed << std::setprecision(2) << fy;
    const cv::Point yTick{toOutputVideoPixel(minX, fy, bounds)};
    cv::putText(
        frame, yLabel.str(), {OUTPUT_VIDEO_MARGIN_LEFT - 60, yTick.y + 4},
        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }

  // Axis labels + title
  cv::putText(frame, "X", {OUTPUT_VIDEO_WIDTH / 2, OUTPUT_VIDEO_HEIGHT - 15},
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1,
              cv::LINE_AA);
  cv::putText(frame, "Y", {15, OUTPUT_VIDEO_HEIGHT / 2},
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1,
              cv::LINE_AA);
  std::ostringstream title;
  title << "t = " << std::fixed << std::setprecision(2) << (frameTimestamp - t0)
        << "s";
  cv::putText(frame, title.str(), {OUTPUT_VIDEO_MARGIN_LEFT, 30},
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1,
              cv::LINE_AA);

  // Legend
  cv::circle(frame, {OUTPUT_VIDEO_WIDTH - 230, 20}, 5, cv::Scalar(0, 160, 0),
             cv::FILLED, cv::LINE_AA);
  cv::putText(frame, "ground truth", {OUTPUT_VIDEO_WIDTH - 215, 25},
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1,
              cv::LINE_AA);
  cv::drawMarker(frame, {OUTPUT_VIDEO_WIDTH - 230, 42}, cv::Scalar(0, 0, 220),
                 cv::MARKER_TILTED_CROSS, 10, 2, cv::LINE_AA);
  cv::putText(frame, "predicted", {OUTPUT_VIDEO_WIDTH - 215, 47},
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1,
              cv::LINE_AA);

  // Ground truth points (and predicted position, if available)
  for (const auto& [trackId, position] : groundTruthPoints) {
    const cv::Point pt{toOutputVideoPixel(position.x, position.y, bounds)};
    cv::circle(frame, pt, 6, cv::Scalar(0, 160, 0), cv::FILLED, cv::LINE_AA);
    cv::putText(frame, std::to_string(trackId), {pt.x + 8, pt.y - 8},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1,
                cv::LINE_AA);

    const auto predIt{predictedPoints.find(trackId)};
    if (predIt != predictedPoints.end()) {
      const cv::Point predPt{
          toOutputVideoPixel(predIt->second.x, predIt->second.y, bounds)};
      cv::line(frame, pt, predPt, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
      cv::drawMarker(frame, predPt, cv::Scalar(0, 0, 220),
                     cv::MARKER_TILTED_CROSS, 12, 2, cv::LINE_AA);
    }
  }

  return frame;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Evaluate the runner lookahead prediction against a rosbag"};
  Options opt;
  app.add_option("--bag", opt.bag, "Path to the detections rosbag directory")
      ->required();
  app.add_option("--topic", opt.topic, "DetectionResult topic to read")
      ->capture_default_str();
  app.add_option(
         "--track-miss-timeout-secs", opt.trackMissTimeoutSecs,
         "Grace period, in seconds, to tolerate a PENDING or ACTIVE track not "
         "appearing in a detection frame before marking it FAILED")
      ->capture_default_str();
  app.add_option(
         "--target-attempts", opt.targetAttempts,
         "Max number of times a FAILED track may be requeued as PENDING after "
         "being redetected. A negative number means no limit")
      ->capture_default_str();
  app.add_option("--lookahead-secs", opt.lookaheadSecs,
                 "How far ahead, in seconds, to predict each track's position")
      ->capture_default_str();
  app.add_flag("--plot,!--no-plot", opt.plot,
               "Render PNG plots and an AVI video of the evaluation results")
      ->capture_default_str();
  app.add_option("--video-fps", opt.videoFps,
                 "Playback frame rate of the output video")
      ->capture_default_str();
  CLI11_PARSE(app, argc, argv);

  // Read the rosbag of DetectionResults
  std::vector<di::DetectionResult> detections;
  try {
    detections = readRunnerDetections(opt.bag, opt.topic);
  } catch (const std::exception& e) {
    std::cerr << "Failed to read bag '" << opt.bag << "': " << e.what() << "\n";
    return 1;
  }
  if (detections.empty()) {
    std::cerr << "No RUNNER DetectionResult messages on " << opt.topic << "\n";
    return 1;
  }

  auto tracker{std::make_shared<Tracker>()};
  DetectionTrackerUpdater updater{tracker,
                                  static_cast<float>(opt.trackMissTimeoutSecs),
                                  opt.targetAttempts};

  // From DetectionResult messages, extract ground truth positions and calculate
  // predicted positions
  std::map<uint32_t, TrackSeries> groundTruths;  // track ID -> TrackSeries
  std::map<uint32_t, TrackSeries> predictions;   // track ID -> TrackSeries
  size_t totalPredictions{0};
  for (const auto& detection : detections) {
    const double detectionTimestampSecs{
        rclcpp::Time(detection.timestamp).seconds()};

    // Extract ground truth for each detection instance
    for (const auto& instance : detection.instances) {
      if (instance.track_id == 0) {
        continue;
      }

      const Position position{static_cast<float>(instance.position.x),
                              static_cast<float>(instance.position.y),
                              static_cast<float>(instance.position.z)};
      if (!validPosition(position)) {
        continue;
      }

      auto& trackSeries{groundTruths[instance.track_id]};
      if (!trackSeries.timestamps.empty() &&
          detectionTimestampSecs <= trackSeries.timestamps.back()) {
        continue;  // ignore out-of-order detections
      }
      trackSeries.timestamps.push_back(detectionTimestampSecs);
      trackSeries.positions.push_back(position);
    }

    // Update the tracker, then for every pending track, predict a lookahead
    // position
    updater.update(detection);
    const double lookaheadTimestampSecs{detectionTimestampSecs +
                                        opt.lookaheadSecs};
    for (const auto& track :
         tracker->getTracksWithState(Track::State::PENDING)) {
      auto& series{predictions[track->getId()]};
      series.timestamps.push_back(lookaheadTimestampSecs);
      series.positions.push_back(
          track->getPredictor().predict(lookaheadTimestampSecs));
      ++totalPredictions;
    }
  }

  // For each ground truth detection, interpolate that track's lookahead
  // prediction to the detection's own timestamp and compare the two
  size_t joinedPredictions{0};
  std::map<uint32_t, std::vector<EvaluatedPrediction>> evaluated;
  for (const auto& [trackId, gtSeries] : groundTruths) {
    const auto predIt{predictions.find(trackId)};
    if (predIt == predictions.end()) {
      continue;
    }

    for (size_t i = 0; i < gtSeries.timestamps.size(); ++i) {
      const double timestamp{gtSeries.timestamps[i]};
      const Position& actual{gtSeries.positions[i]};
      const auto predicted{interpolateAt(predIt->second, timestamp)};
      if (!predicted) {
        continue;  // no lookahead prediction spans this detection's timestamp
      }

      ++joinedPredictions;
      evaluated[trackId].push_back(
          {timestamp, *predicted, actual, distance(*predicted, actual)});
    }
  }

  // Print summary
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "evaluate_lookahead\n"
            << "  bag: " << opt.bag << " (topic " << opt.topic << ")\n"
            << "  # frames: " << detections.size() << "\n"
            << "  # tracks: " << groundTruths.size() << "\n"
            << "  # predictions made: " << totalPredictions << "\n"
            << "  # ground truth detections evaluated: " << joinedPredictions
            << "\n";
  for (const auto& [trackId, evs] : evaluated) {
    double totalError{0.0};
    for (const auto& ev : evs) {
      totalError += ev.error;
    }
    std::cout << "  track " << trackId << ": " << evs.size()
              << " detections evaluated, mean error "
              << (evs.empty() ? 0.0
                              : totalError / static_cast<double>(evs.size()))
              << "\n";
  }

  if (evaluated.empty()) {
    std::cerr << "No predictions could be evaluated.\n";
    return 1;
  }

  if (opt.plot) {
    // Render PNGs headlessly
    plt::backend("Agg");

    // Plot error vs time, one line per track
    const double t0{rclcpp::Time(detections.front().timestamp).seconds()};
    plt::figure();
    for (const auto& [trackId, evs] : evaluated) {
      std::vector<double> x;
      std::vector<double> y;
      x.reserve(evs.size());
      y.reserve(evs.size());
      for (const auto& ev : evs) {
        x.push_back(ev.timestamp - t0);
        y.push_back(ev.error);
      }
      plt::named_plot("track " + std::to_string(trackId), x, y);
    }
    plt::xlabel("time (s)");
    plt::ylabel("lookahead position error");
    plt::title("Lookahead prediction error vs time");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    const std::string errorOverTimePlotFilename{
        "evaluate_lookahead_error_vs_time.png"};
    plt::save(errorOverTimePlotFilename);
    plt::close();
    std::cout << "wrote " << errorOverTimePlotFilename << "\n";

    // Render one video frame per DetectionResult. Each frame is an X-Y plot of
    // ground truth positions of detected tracks, plus each track's lookahead
    // prediction (if available) interpolated to the frame's timestamp.
    const std::string videoFilename{"evaluate_lookahead.avi"};
    cv::VideoWriter writer{
        videoFilename, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        opt.videoFps, cv::Size(OUTPUT_VIDEO_WIDTH, OUTPUT_VIDEO_HEIGHT)};
    if (!writer.isOpened()) {
      std::cerr << "Failed to open " << videoFilename << " for writing\n";
    } else {
      const cv::Rect2d bounds{computeBounds(groundTruths, predictions)};
      for (const auto& detection : detections) {
        const double detectionTimestampSecs{
            rclcpp::Time(detection.timestamp).seconds()};

        std::vector<std::pair<uint32_t, Position>> groundTruthPoints;
        for (const auto& instance : detection.instances) {
          if (instance.track_id == 0) {
            continue;
          }
          const Position position{static_cast<float>(instance.position.x),
                                  static_cast<float>(instance.position.y),
                                  static_cast<float>(instance.position.z)};
          if (!validPosition(position)) {
            continue;
          }
          groundTruthPoints.emplace_back(instance.track_id, position);
        }

        std::map<uint32_t, Position> predictedPoints;
        for (const auto& [trackId, position] : groundTruthPoints) {
          (void)position;
          const auto seriesIt{predictions.find(trackId)};
          if (seriesIt == predictions.end()) {
            continue;
          }
          const auto predicted{
              interpolateAt(seriesIt->second, detectionTimestampSecs)};
          if (predicted) {
            predictedPoints[trackId] = *predicted;
          }
        }

        writer.write(renderOutputVideoFrame(detectionTimestampSecs, t0,
                                            groundTruthPoints, predictedPoints,
                                            bounds));
      }
      writer.release();
      std::cout << "wrote " << videoFilename << " (" << detections.size()
                << " frames)\n";
    }
  }

  return 0;
}
