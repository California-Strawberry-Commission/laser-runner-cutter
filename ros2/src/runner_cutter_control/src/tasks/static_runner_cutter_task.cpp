#include "runner_cutter_control/tasks/static_runner_cutter_task.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "common/ros_utils.hpp"
#include "common_interfaces/msg/vector2.hpp"
#include "detection_interfaces/msg/detection_result.hpp"
#include "detection_interfaces/msg/detection_type.hpp"
#include "runner_cutter_control/tasks/detection_tracker_updater.hpp"
#include "runner_cutter_control_interfaces/msg/track.hpp"
#include "runner_cutter_control_interfaces/msg/track_state.hpp"

StaticRunnerCutterTask::StaticRunnerCutterTask(
    std::shared_ptr<
        CallbackRegistry<detection_interfaces::msg::DetectionResult>>
        detectionCallbackRegistry,
    std::shared_ptr<CameraControlClient> camera,
    std::shared_ptr<DetectionClient> detection,
    std::shared_ptr<Calibration> calibration,
    std::shared_ptr<LaserControlClient> laser, rclcpp::Logger logger,
    rclcpp::Publisher<rcl_interfaces::msg::Log>::SharedPtr
        notificationsPublisher,
    rclcpp::Publisher<runner_cutter_control_interfaces::msg::Tracks>::SharedPtr
        tracksPublisher)
    : detectionCallbackRegistry_(std::move(detectionCallbackRegistry)),
      camera_(std::move(camera)),
      detection_(std::move(detection)),
      calibration_(std::move(calibration)),
      laserTargeting_(std::move(laser), camera_, detection_, calibration_,
                      logger),
      logger_(std::move(logger)),
      notificationsPublisher_(std::move(notificationsPublisher)),
      tracksPublisher_(std::move(tracksPublisher)),
      tracker_(std::make_shared<Tracker>()) {}

void StaticRunnerCutterTask::run(float trackMissTimeoutSecs, int targetAttempts,
                                 bool enableDetectionDuringBurn,
                                 bool enableAiming, float autoDisarmSecs,
                                 const std::string& saveDir,
                                 const LaserColor& trackingLaserColor,
                                 const LaserColor& burnLaserColor,
                                 float burnTimeSecs,
                                 std::atomic<bool>& stopSignal) {
  // Save image to saveDir
  auto timestamp{
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
  std::stringstream datetimeString;
  datetimeString << std::put_time(std::localtime(&timestamp), "%Y%m%d%H%M%S");
  std::string runDataDir{
      fmt::format("{}/runs/{}", saveDir, datetimeString.str())};
  camera_->setSaveDirectory(runDataDir);
  camera_->saveImage();

  tracker_->clear();
  DetectionTrackerUpdater updater{tracker_, trackMissTimeoutSecs,
                                  targetAttempts};
  publishTracks();

  // Register to receive per-frame detection updates during the task
  detectionCallbackRegistry_->set(
      [this, &updater,
       &stopSignal](detection_interfaces::msg::DetectionResult::SharedPtr msg) {
        // Only process RUNNER detections
        if (stopSignal ||
            msg->detection_type !=
                detection_interfaces::msg::DetectionType::RUNNER) {
          return;
        }

        // Update the tracker
        if (updater.update(*msg)) {
          pendingTracksChangedEvent_.set();
        }
      });

  // Start runner detection with detection bounds set to the laser's FOV.
  // Note: The ML model will still detect runners and assign instance IDs
  // using the full color camera frame, but if the runner is completely out of
  // the detection bounds, the result is not published via the detections
  // topic.
  NormalizedPixelRect normalizedLaserBounds{
      calibration_->getNormalizedLaserBounds()};
  detection_->startDetection(detection_interfaces::msg::DetectionType::RUNNER,
                             normalizedLaserBounds);

  RCLCPP_INFO(logger_,
              "Runner cutter ARMED: normalizedLaserBounds=[u=%f, v=%f, "
              "width=%f, height=%f]",
              normalizedLaserBounds.u, normalizedLaserBounds.v,
              normalizedLaserBounds.width, normalizedLaserBounds.height);

  while (!stopSignal) {
    // Attempt to acquire target
    auto targetOpt{acquireNextTarget()};
    publishTracks();

    // If there are no valid targets, wait until pending tracks changes
    if (!targetOpt) {
      RCLCPP_INFO(
          logger_,
          "No suitable targets found. Waiting for more pending targets.");
      bool pendingTracksChanged{
          waitForPendingTracks(autoDisarmSecs, stopSignal)};
      pendingTracksChangedEvent_.clear();
      if (!pendingTracksChanged) {
        if (stopSignal) {
          // Stop signal triggered. End task.
          break;
        }

        // No new valid targets for autoDisarmSecs. End task.
        common::publishNotification(
            logger_, notificationsPublisher_,
            fmt::format(
                "No new targets after {} second(s). Ending runner cutter "
                "task.",
                autoDisarmSecs));
        break;
      }

      // A new pending track was detected. Go to the next iteration to attempt
      // to acquire a target again.
      continue;
    }

    // Temporarily disable runner detection during aim/burn if needed
    if (!enableDetectionDuringBurn) {
      detection_->stopDetection(
          detection_interfaces::msg::DetectionType::RUNNER);
    }

    auto target{std::move(*targetOpt)};

    // Aim
    LaserCoord laserCoord;
    if (enableAiming) {
      auto laserCoordOpt{laserTargeting_.aim(
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
    laserTargeting_.burn(target->getId(), laserCoord, burnLaserColor,
                         burnTimeSecs);
    tracker_->transitionTrackState(target->getId(), Track::State::COMPLETED);

    // Re-enable runner detection after burn if needed
    if (!enableDetectionDuringBurn) {
      detection_->startDetection(
          detection_interfaces::msg::DetectionType::RUNNER,
          normalizedLaserBounds);
    }
  }

  detection_->stopAllDetections();
  detectionCallbackRegistry_->clear();
}

std::optional<std::shared_ptr<const Track>>
StaticRunnerCutterTask::acquireNextTarget() {
  // If there is already an active track, return the first one
  auto activeTracks{tracker_->getTracksWithState(Track::State::ACTIVE)};
  if (!activeTracks.empty()) {
    RCLCPP_INFO(logger_, "Using active track %u", activeTracks[0]->getId());
    return activeTracks[0];
  }

  // There are no active tracks. Attempt to activate a pending track, and if it
  // is within laser bounds, return it. Otherwise, mark it as failed and try
  // another one.
  while (true) {
    auto trackOpt{tracker_->activateNextPendingTrack()};
    if (!trackOpt) {
      return std::nullopt;
    }

    auto track{std::move(*trackOpt)};
    LaserCoord laserCoord{
        calibration_->cameraPositionToLaserCoord(track->getPosition())};
    if (laserCoord.x >= 0.0 && laserCoord.x <= 1.0 && laserCoord.y >= 0.0 &&
        laserCoord.y <= 1.0) {
      RCLCPP_INFO(logger_, "Setting track %u as target.", track->getId());
      return track;
    }

    RCLCPP_INFO(logger_, "Track %u is out of laser bounds. Marking as failed.",
                track->getId());
    tracker_->transitionTrackState(track->getId(), Track::State::FAILED);
  }

  return std::nullopt;
}

runner_cutter_control_interfaces::msg::Tracks::UniquePtr
StaticRunnerCutterTask::getTracksMsg() {
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

void StaticRunnerCutterTask::publishTracks() {
  tracksPublisher_->publish(std::move(getTracksMsg()));
}

bool StaticRunnerCutterTask::waitForPendingTracks(
    float timeoutSecs, std::atomic<bool>& stopSignal) {
  bool hasDeadline{timeoutSecs > 0.0f};
  auto deadline{std::chrono::steady_clock::now() +
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<float>(timeoutSecs))};

  // Poll in short increments so stopSignal is checked promptly
  while (!stopSignal) {
    float pollSecs{0.05f};
    if (hasDeadline) {
      float remainingSecs{std::chrono::duration<float>(
                              deadline - std::chrono::steady_clock::now())
                              .count()};
      if (remainingSecs <= 0.0f) {
        return false;
      }
      pollSecs = std::min(remainingSecs, pollSecs);
    }

    if (pendingTracksChangedEvent_.wait_for(pollSecs)) {
      return true;
    }
  }

  return false;
}
