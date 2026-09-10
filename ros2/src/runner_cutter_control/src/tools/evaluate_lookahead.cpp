// evaluate_lookahead
//
// Offline evaluation of lookahead prediction against a rosbag of
// DetectionResult messages.
//
//   1. Read the rosbag of DetectionResult messages.
//   2. Extract the ground-truth position time series for each track.
//   3. Replay every DetectionResult through DetectionTrackerUpdater and, each
//      frame, ask every track's predictor where it will be X ms in the future.
//   4. Compare each prediction against the position that track is actually
//      detected at that future time (linear interpolation of its own
//      detections) to calculate the error.

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "CLI/CLI.hpp"
#include "camera_control/utils/rgbd_alignment.hpp"
#include "detection_interfaces/msg/detection_result.hpp"
#include "detection_interfaces/msg/detection_type.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "matplotlibcpp.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/serialization.hpp"
#include "rclcpp/serialized_message.hpp"
#include "rclcpp/time.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "runner_cutter_control/common_types.hpp"
#include "runner_cutter_control/tasks/detection_tracker_updater.hpp"
#include "runner_cutter_control/tracking/track.hpp"
#include "runner_cutter_control/tracking/tracker.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2_msgs/msg/tf_message.hpp"

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
  std::string cameraBag;
  std::string cameraTopic{"/camera0/color/image_raw"};
  std::string cameraInfoTopic{"/camera0/color/camera_info"};
  std::string outputVideo{"evaluate_lookahead.avi"};
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
  std::vector<PixelCoord> pixelCoords;
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

// Finds the pixel coord in `series` whose timestamp is closest to
// `queryTimestamp`. Returns nullopt if `series` is empty.
std::optional<PixelCoord> nearestPixelAt(const TrackSeries& series,
                                         double queryTimestamp) {
  if (series.timestamps.empty()) {
    return std::nullopt;
  }
  const auto next{std::lower_bound(series.timestamps.begin(),
                                   series.timestamps.end(), queryTimestamp)};
  if (next == series.timestamps.begin()) {
    return series.pixelCoords.front();
  }
  if (next == series.timestamps.end()) {
    return series.pixelCoords.back();
  }
  const size_t nextIdx{static_cast<size_t>(next - series.timestamps.begin())};
  const size_t prevIdx{nextIdx - 1};
  const double prevDelta{std::abs(series.timestamps[prevIdx] - queryTimestamp)};
  const double nextDelta{std::abs(series.timestamps[nextIdx] - queryTimestamp)};
  return (prevDelta <= nextDelta) ? series.pixelCoords[prevIdx]
                                  : series.pixelCoords[nextIdx];
}

double distance(const Position& a, const Position& b) {
  const double dx{static_cast<double>(a.x) - b.x};
  const double dy{static_cast<double>(a.y) - b.y};
  const double dz{static_cast<double>(a.z) - b.z};
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

struct Prediction {
  double frameTimestamp{0.0};
  double lookaheadTimestamp{0.0};
  Position predicted{};
};

struct EvaluatedPrediction {
  double frameTimestamp{0.0};
  double lookaheadTimestamp{0.0};
  Position predicted{};
  Position actual{};
  double error{0.0};
};

// Linear interpolation of `preds`' predicted positions at `queryTimestamp`,
// using each prediction's lookaheadTimestamp. Returns nullopt if
// `queryTimestamp` is outside the predictions' time span or there are fewer
// than two predictions. `preds` is assumed to be sorted ascending by
// lookaheadTimestamp.
std::optional<Position> interpolatePredictedAt(
    const std::vector<Prediction>& preds, double queryTimestamp) {
  if (preds.size() < 2 || queryTimestamp < preds.front().lookaheadTimestamp ||
      queryTimestamp > preds.back().lookaheadTimestamp) {
    return std::nullopt;
  }
  const auto hi{std::upper_bound(preds.begin(), preds.end(), queryTimestamp,
                                 [](double timestamp, const Prediction& pred) {
                                   return timestamp < pred.lookaheadTimestamp;
                                 })};
  if (hi == preds.end()) {
    return preds.back().predicted;
  }
  const size_t i2{static_cast<size_t>(hi - preds.begin())};
  const size_t i1{i2 - 1};
  const double a{(queryTimestamp - preds[i1].lookaheadTimestamp) /
                 (preds[i2].lookaheadTimestamp - preds[i1].lookaheadTimestamp)};
  const Position& p1{preds[i1].predicted};
  const Position& p2{preds[i2].predicted};
  return Position{static_cast<float>(p1.x + a * (p2.x - p1.x)),
                  static_cast<float>(p1.y + a * (p2.y - p1.y)),
                  static_cast<float>(p1.z + a * (p2.z - p1.z))};
}

// Converts a tf transform into a 4x4 extrinsic matrix
cv::Mat toExtrinsicMatrix(const geometry_msgs::msg::Transform& transform) {
  const Eigen::Quaterniond rotation{transform.rotation.w, transform.rotation.x,
                                    transform.rotation.y, transform.rotation.z};
  const Eigen::Matrix3d rotationMatrix{
      rotation.normalized().toRotationMatrix()};

  cv::Mat extrinsic{cv::Mat::eye(4, 4, CV_64F)};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      extrinsic.at<double>(r, c) = rotationMatrix(r, c);
    }
  }
  extrinsic.at<double>(0, 3) = transform.translation.x;
  extrinsic.at<double>(1, 3) = transform.translation.y;
  extrinsic.at<double>(2, 3) = transform.translation.z;
  return extrinsic;
}

cv::Mat toIntrinsicMatrix(const sensor_msgs::msg::CameraInfo& cameraInfo) {
  cv::Mat k(3, 3, CV_64F);
  for (int i = 0; i < 9; ++i) {
    k.at<double>(i / 3, i % 3) = cameraInfo.k[i];
  }
  return k;
}

cv::Mat toDistCoeffs(const sensor_msgs::msg::CameraInfo& cameraInfo) {
  cv::Mat d(1, static_cast<int>(cameraInfo.d.size()), CV_64F);
  for (size_t i = 0; i < cameraInfo.d.size(); ++i) {
    d.at<double>(0, static_cast<int>(i)) = cameraInfo.d[i];
  }
  return d;
}

// Demosaics a raw BAYER_RGGB8 color frame into a BGR image.
cv::Mat toBgrMat(const sensor_msgs::msg::Image& image) {
  cv::Mat raw(static_cast<int>(image.height), static_cast<int>(image.width),
              CV_8UC1, const_cast<uint8_t*>(image.data.data()), image.step);
  cv::Mat bgr;
  cv::cvtColor(raw, bgr, cv::COLOR_BayerRG2BGR);
  return bgr;
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
               "Render PNG plots of the evaluation results")
      ->capture_default_str();
  app.add_option(
      "--camera-bag", opt.cameraBag,
      "Path to a rosbag of color camera images, camera_info, and /tf_static. "
      "If provided, for each frame, draws each track's projected predicted "
      "position and ground-truth pixel and writes the annotated frames to "
      "--output-video");
  app.add_option("--camera-topic", opt.cameraTopic,
                 "Color image topic to read from --camera-bag")
      ->capture_default_str();
  app.add_option("--camera-info-topic", opt.cameraInfoTopic,
                 "Color CameraInfo topic to read from --camera-bag")
      ->capture_default_str();
  app.add_option("--output-video", opt.outputVideo,
                 "Path to write the annotated camera frames to, as a video")
      ->capture_default_str();
  app.add_option("--video-fps", opt.videoFps, "Frame rate of the output video")
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

  std::map<uint32_t, TrackSeries> groundTruth;
  std::map<uint32_t, std::vector<Prediction>> predictions;
  for (const auto& detection : detections) {
    const double timestamp{rclcpp::Time(detection.timestamp).seconds()};

    // Extract ground truth for each instance
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

      auto& trackSeries{groundTruth[instance.track_id]};
      if (!trackSeries.timestamps.empty() &&
          timestamp <= trackSeries.timestamps.back()) {
        continue;  // ignore out-of-order detections
      }
      trackSeries.timestamps.push_back(timestamp);
      trackSeries.positions.push_back(position);
      trackSeries.pixelCoords.push_back(
          PixelCoord{static_cast<int>(std::round(instance.point.x)),
                     static_cast<int>(std::round(instance.point.y))});
    }

    // Update the tracker, then for every pending track, predict the lookahead
    // position
    updater.update(detection);
    const double lookaheadTimestamp{timestamp + opt.lookaheadSecs};
    for (const auto& track :
         tracker->getTracksWithState(Track::State::PENDING)) {
      predictions[track->getId()].push_back(
          {timestamp, lookaheadTimestamp,
           track->getPredictor().predict(lookaheadTimestamp)});
    }
  }

  // For each lookahead prediction made, find the ground truth position at the
  // lookahead timestamp
  std::map<uint32_t, std::vector<EvaluatedPrediction>> evaluated;
  size_t totalPredictions{0};
  size_t joinedPredictions{0};
  for (const auto& [trackId, preds] : predictions) {
    const auto gtIt{groundTruth.find(trackId)};
    if (gtIt == groundTruth.end()) {
      continue;
    }

    for (const auto& pred : preds) {
      ++totalPredictions;
      const auto actual{interpolateAt(gtIt->second, pred.lookaheadTimestamp)};
      if (!actual) {
        continue;  // lookahead time is beyond this track's detection span
      }

      ++joinedPredictions;
      evaluated[trackId].push_back(
          {pred.frameTimestamp, pred.lookaheadTimestamp, pred.predicted,
           *actual, distance(pred.predicted, *actual)});
    }
  }

  // Print summary
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "evaluate_lookahead\n"
            << "  bag: " << opt.bag << " (topic " << opt.topic << ")\n"
            << "  # frames: " << detections.size() << "\n"
            << "  # tracks: " << groundTruth.size() << "\n"
            << "  # predictions made: " << totalPredictions << " ("
            << joinedPredictions << " evaluated)\n";
  for (const auto& [trackId, evs] : evaluated) {
    double totalError{0.0};
    for (const auto& ev : evs) {
      totalError += ev.error;
    }
    std::cout << "  track " << trackId << ": " << evs.size()
              << " predictions evaluated, mean error "
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
        x.push_back(ev.frameTimestamp - t0);
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

    // Predicted vs actual X-Y trajectory at the lookahead time, one plot per
    // track
    for (const auto& [trackId, evs] : evaluated) {
      std::vector<double> px;
      std::vector<double> py;
      std::vector<double> ax;
      std::vector<double> ay;
      px.reserve(evs.size());
      py.reserve(evs.size());
      ax.reserve(evs.size());
      ay.reserve(evs.size());
      for (const auto& ev : evs) {
        px.push_back(ev.predicted.x);
        py.push_back(ev.predicted.y);
        ax.push_back(ev.actual.x);
        ay.push_back(ev.actual.y);
      }

      plt::figure();
      plt::named_plot("predicted", px, py, "r.-");
      plt::named_plot("actual", ax, ay, "g.-");
      plt::xlabel("X");
      plt::ylabel("Y");
      plt::title("Track " + std::to_string(trackId) +
                 " -- predicted vs actual at +" +
                 std::to_string(opt.lookaheadSecs) + "s");
      plt::legend();
      plt::grid(true);
      plt::set_aspect_equal();
      plt::tight_layout();
      const std::string trackPlotFilename{"evaluate_lookahead_track_" +
                                          std::to_string(trackId) + ".png"};
      plt::save(trackPlotFilename);
      plt::close();
      std::cout << "wrote " << trackPlotFilename << "\n";
    }
  }

  // If a camera bag was provided, stream color images one frame at a time,
  // picking up the color camera's intrinsic matrix, distortion coefficients,
  // and XYZ to color extrinsic transform. For each frame, for every track,
  // linearly interpolate the predicted position at the frame's timestamp and
  // draw it, alongside the ground-truth nearest that same timestamp.
  if (!opt.cameraBag.empty()) {
    try {
      rosbag2_cpp::Reader reader;
      reader.open(opt.cameraBag);

      const std::filesystem::path outputVideoPath{opt.outputVideo};
      if (outputVideoPath.has_parent_path()) {
        std::filesystem::create_directories(outputVideoPath.parent_path());
      }
      const cv::Size outputFrameSize{1024, 768};
      cv::VideoWriter videoWriter{opt.outputVideo,
                                  cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                                  opt.videoFps, outputFrameSize};
      if (!videoWriter.isOpened()) {
        std::cerr << "Failed to open output video '" << opt.outputVideo
                  << "'\n";
        return 1;
      }

      rclcpp::Serialization<sensor_msgs::msg::Image> imageSerialization;
      rclcpp::Serialization<sensor_msgs::msg::CameraInfo>
          cameraInfoSerialization;
      rclcpp::Serialization<tf2_msgs::msg::TFMessage> tfSerialization;

      std::optional<cv::Mat> colorCameraIntrinsicMatrix;
      std::optional<cv::Mat> colorCameraDistCoeffs;
      std::optional<cv::Mat> xyzToColorCameraExtrinsicMatrix;

      while (reader.has_next()) {
        const auto bagMsg{reader.read_next()};
        rclcpp::SerializedMessage serialized{*bagMsg->serialized_data};

        // Extract the XYZ to color camera extrinsic matrix
        if (bagMsg->topic_name == "/tf_static") {
          if (!xyzToColorCameraExtrinsicMatrix) {
            tf2_msgs::msg::TFMessage tfMsg;
            tfSerialization.deserialize_message(&serialized, &tfMsg);
            for (const auto& transform : tfMsg.transforms) {
              if (transform.child_frame_id == "color_camera") {
                xyzToColorCameraExtrinsicMatrix =
                    toExtrinsicMatrix(transform.transform);
                break;
              }
            }
          }
          continue;
        }

        // Extract camera intrinsic matrix and distortion coeffs
        if (bagMsg->topic_name == opt.cameraInfoTopic) {
          if (!colorCameraIntrinsicMatrix) {
            sensor_msgs::msg::CameraInfo cameraInfo;
            cameraInfoSerialization.deserialize_message(&serialized,
                                                        &cameraInfo);
            colorCameraIntrinsicMatrix = toIntrinsicMatrix(cameraInfo);
            colorCameraDistCoeffs = toDistCoeffs(cameraInfo);
          }
          continue;
        }

        if (bagMsg->topic_name != opt.cameraTopic) {
          continue;
        }

        sensor_msgs::msg::Image frame;
        imageSerialization.deserialize_message(&serialized, &frame);

        const double frameTimestamp{rclcpp::Time(frame.header.stamp).seconds()};
        cv::Mat image;
        try {
          image = toBgrMat(frame);
        } catch (const std::exception& e) {
          std::cerr << "Failed to convert frame at " << frameTimestamp << ": "
                    << e.what() << "\n";
          continue;
        }

        if (colorCameraIntrinsicMatrix && xyzToColorCameraExtrinsicMatrix) {
          for (const auto& [trackId, preds] : predictions) {
            const auto predicted{interpolatePredictedAt(preds, frameTimestamp)};
            if (!predicted) {
              continue;
            }

            // Predicted position, projected onto the color frame
            const cv::Vec3f predictedPosition{predicted->x, predicted->y,
                                              predicted->z};
            const auto predictedPixel{RgbdAlignment::projectPosition(
                predictedPosition, *colorCameraIntrinsicMatrix,
                *colorCameraDistCoeffs, *xyzToColorCameraExtrinsicMatrix)};
            if (predictedPixel) {
              const cv::Point center{predictedPixel->x, predictedPixel->y};
              cv::circle(image, center, 8, cv::Scalar(0, 0, 255), 2);
              cv::putText(image, "predicted " + std::to_string(trackId),
                          center + cv::Point{10, -10}, cv::FONT_HERSHEY_SIMPLEX,
                          0.5, cv::Scalar(0, 0, 255), 2);
            }

            // Ground-truth pixel nearest the frame's timestamp
            const auto gtIt{groundTruth.find(trackId)};
            if (gtIt != groundTruth.end()) {
              const auto actualPixel{
                  nearestPixelAt(gtIt->second, frameTimestamp)};
              if (actualPixel) {
                const cv::Point center{actualPixel->u, actualPixel->v};
                cv::circle(image, center, 8, cv::Scalar(0, 255, 0), 2);
                cv::putText(image, "actual " + std::to_string(trackId),
                            center + cv::Point{10, 10},
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 0), 2);
              }
            }
          }
        }

        cv::resize(image, image, outputFrameSize);
        videoWriter.write(image);
      }
      reader.close();
      videoWriter.release();
      std::cout << "wrote " << opt.outputVideo << "\n";
    } catch (const std::exception& e) {
      std::cerr << "Failed to read camera bag '" << opt.cameraBag
                << "': " << e.what() << "\n";
      return 1;
    }
  }

  return 0;
}
