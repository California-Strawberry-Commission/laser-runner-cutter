#include "runner_cutter_control/tasks/runner_cutter_task.hpp"

#include <fmt/core.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "common/ros_utils.hpp"
#include "common_interfaces/msg/vector2.hpp"
#include "detection_interfaces/msg/detection_type.hpp"
#include "runner_cutter_control_interfaces/msg/track.hpp"
#include "runner_cutter_control_interfaces/msg/track_state.hpp"

RunnerCutterTask::RunnerCutterTask(
    std::shared_ptr<CameraControlClient> camera,
    std::shared_ptr<DetectionClient> detection,
    std::shared_ptr<Calibration> calibration, std::shared_ptr<Tracker> tracker,
    std::shared_ptr<LaserTargeting> laserTargeting, rclcpp::Logger logger,
    rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
        notificationsPublisher,
    rclcpp::Publisher<runner_cutter_control_interfaces::msg::Tracks>::SharedPtr
        tracksPublisher)
    : camera_(std::move(camera)),
      detection_(std::move(detection)),
      calibration_(std::move(calibration)),
      tracker_(std::move(tracker)),
      laserTargeting_(std::move(laserTargeting)),
      logger_(std::move(logger)),
      notificationsPublisher_(std::move(notificationsPublisher)),
      tracksPublisher_(std::move(tracksPublisher)) {}

void RunnerCutterTask::run(uint8_t detectionType,
                           bool enableDetectionDuringBurn, bool enableAiming,
                           float autoDisarmSecs, const std::string& saveDir,
                           const LaserColor& trackingLaserColor,
                           const LaserColor& burnLaserColor, float burnTimeSecs,
                           std::atomic<bool>& stopSignal,
                           common::Event& pendingTracksChangedEvent) {
  publishTracks();

  auto timestamp{
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
  std::stringstream datetimeString;
  datetimeString << std::put_time(std::localtime(&timestamp), "%Y%m%d%H%M%S");
  std::string runDataDir{
      fmt::format("{}/runs/{}", saveDir, datetimeString.str())};
  camera_->setSaveDirectory(runDataDir);
  camera_->saveImage();

  // Start runner detection with detection bounds set to the laser's FOV.
  // Note: The ML model will still detect runners and assign instance IDs
  // using the full color camera frame, but if the runner is completely out of
  // the detection bounds, the result is not published via the detections
  // topic.
  NormalizedPixelRect normalizedLaserBounds{
      calibration_->getNormalizedLaserBounds()};
  RCLCPP_INFO(logger_,
              "Runner cutter ARMED: detectionType=%d, "
              "normalizedLaserBounds=[u=%f, v=%f, width=%f, height=%f]",
              static_cast<int>(detectionType), normalizedLaserBounds.u,
              normalizedLaserBounds.v, normalizedLaserBounds.width,
              normalizedLaserBounds.height);
  detection_->startDetection(detectionType, normalizedLaserBounds);

  while (!stopSignal) {
    // Attempt to acquire target
    auto targetOpt{acquireNextTarget(stopSignal)};
    publishTracks();

    // If there are no valid targets, wait for another detection event
    if (!targetOpt) {
      RCLCPP_INFO(logger_, "No targets found. Waiting for detection.");
      if (autoDisarmSecs > 0.0f) {
        // End task if no new valid targets for autoDisarmSecs
        if (!pendingTracksChangedEvent.wait_for(autoDisarmSecs)) {
          common::publishNotification(
              logger_, notificationsPublisher_,
              fmt::format(
                  "No new targets after {} second(s). Ending runner cutter "
                  "task.",
                  autoDisarmSecs));
          break;
        }

      } else {
        pendingTracksChangedEvent.wait();
      }
      pendingTracksChangedEvent.clear();
      continue;
    }

    // Temporarily disable runner detection during aim/burn if needed
    if (!enableDetectionDuringBurn) {
      detection_->stopDetection(detectionType);
    }

    auto target{std::move(*targetOpt)};

    // Aim
    LaserCoord laserCoord;
    if (enableAiming) {
      auto laserCoordOpt{laserTargeting_->aim(
          target->getId(), target->getPosition(), target->getPixel(),
          trackingLaserColor, stopSignal)};
      if (!laserCoordOpt) {
        RCLCPP_INFO(logger_, "Failed to aim laser at track %u.",
                    target->getId());
        tracker_->transitionTrackState(target->getId(), Track::State::FAILED);
        continue;
      }
      laserCoord = std::move(*laserCoordOpt);
    } else {
      laserCoord =
          calibration_->cameraPositionToLaserCoord(target->getPosition());
    }

    // Burn
    laserTargeting_->burn(target->getId(), laserCoord, burnLaserColor,
                          burnTimeSecs);
    tracker_->transitionTrackState(target->getId(), Track::State::COMPLETED);

    // Re-enable runner detection after burn if needed
    if (!enableDetectionDuringBurn) {
      detection_->startDetection(detectionType, normalizedLaserBounds);
    }
  }
}

std::optional<std::shared_ptr<const Track>> RunnerCutterTask::acquireNextTarget(
    std::atomic<bool>& stopSignal) {
  auto activeTracks{tracker_->getTracksWithState(Track::State::ACTIVE)};
  if (!activeTracks.empty()) {
    RCLCPP_INFO(logger_, "Using active track %u", activeTracks[0]->getId());
    return activeTracks[0];
  }

  while (!stopSignal) {
    auto trackOpt{tracker_->activateNextPendingTrack()};
    if (!trackOpt) {
      return std::nullopt;
    }

    auto track{std::move(*trackOpt)};
    RCLCPP_INFO(logger_, "Processing pending track %u", track->getId());

    LaserCoord laserCoord{
        calibration_->cameraPositionToLaserCoord(track->getPosition())};
    if (laserCoord.x >= 0.0 && laserCoord.x <= 1.0 && laserCoord.y >= 0.0 &&
        laserCoord.y <= 1.0) {
      RCLCPP_INFO(logger_, "Setting track %u as target.", track->getId());
      return track;
    }

    RCLCPP_INFO(logger_, "Track %u out of bounds. Marking as failed.",
                track->getId());
    tracker_->transitionTrackState(track->getId(), Track::State::FAILED);
  }

  return std::nullopt;
}

runner_cutter_control_interfaces::msg::Tracks::UniquePtr
RunnerCutterTask::getTracksMsg() {
  auto msg{std::make_unique<runner_cutter_control_interfaces::msg::Tracks>()};
  auto [frameWidth, frameHeight]{calibration_->getCameraFrameSize()};
  for (const auto& [id, track] : tracker_->getTracks()) {
    runner_cutter_control_interfaces::msg::Track trackMsg;
    trackMsg.id = track->getId();
    common_interfaces::msg::Vector2 normalizedPixelCoordMsg;
    normalizedPixelCoordMsg.x = frameWidth > 0
                                    ? static_cast<float>(track->getPixel().u) /
                                          static_cast<float>(frameWidth)
                                    : -1.0f;
    normalizedPixelCoordMsg.y = frameHeight > 0
                                    ? static_cast<float>(track->getPixel().v) /
                                          static_cast<float>(frameHeight)
                                    : -1.0f;
    trackMsg.normalized_pixel_coord = normalizedPixelCoordMsg;
    switch (track->getState()) {
      case Track::State::PENDING:
        trackMsg.state =
            runner_cutter_control_interfaces::msg::TrackState::PENDING;
        break;
      case Track::State::ACTIVE:
        trackMsg.state =
            runner_cutter_control_interfaces::msg::TrackState::ACTIVE;
        break;
      case Track::State::COMPLETED:
        trackMsg.state =
            runner_cutter_control_interfaces::msg::TrackState::COMPLETED;
        break;
      case Track::State::FAILED:
        trackMsg.state =
            runner_cutter_control_interfaces::msg::TrackState::FAILED;
        break;
    }
    msg->tracks.push_back(trackMsg);
  }
  return msg;
}

void RunnerCutterTask::publishTracks() {
  tracksPublisher_->publish(std::move(getTracksMsg()));
}
