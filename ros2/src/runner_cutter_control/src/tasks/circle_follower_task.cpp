#include "runner_cutter_control/tasks/circle_follower_task.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#include "detection_interfaces/msg/detection_type.hpp"
#include "runner_cutter_control/tasks/detection_tracker_updater.hpp"
#include "runner_cutter_control/tracking/tracker.hpp"

CircleFollowerTask::CircleFollowerTask(
    std::shared_ptr<
        CallbackRegistry<detection_interfaces::msg::DetectionResult>>
        detectionCallbackRegistry,
    std::shared_ptr<LaserControlClient> laser,
    std::shared_ptr<DetectionClient> detection,
    std::shared_ptr<Calibration> calibration, rclcpp::Logger logger)
    : detectionCallbackRegistry_(std::move(detectionCallbackRegistry)),
      laser_(std::move(laser)),
      detection_(std::move(detection)),
      calibration_(std::move(calibration)),
      logger_(std::move(logger)) {}

void CircleFollowerTask::run(float trackMissTimeoutSecs, int targetAttempts,
                             float lookaheadSecs, const LaserColor& laserColor,
                             float laserIntervalSecs,
                             std::atomic<bool>& stopSignal) {
  auto tracker{std::make_shared<Tracker>()};
  DetectionTrackerUpdater updater{tracker, trackMissTimeoutSecs,
                                  targetAttempts};

  laser_->clearPaths();
  laser_->setColor(LaserColor{0.0f, 0.0f, 0.0f, 0.0f});

  // Arm laser
  laser_->play();

  // Register to receive per-frame detection updates during the task
  detectionCallbackRegistry_->set(
      [this, lookaheadSecs, tracker, &updater,
       &stopSignal](detection_interfaces::msg::DetectionResult::SharedPtr msg) {
        // Only process test circle detections
        if (stopSignal ||
            msg->detection_type !=
                detection_interfaces::msg::DetectionType::CIRCLE) {
          return;
        }

        // Circle tracking is for testing purposes. For now, if there are any
        // detection object instances, just take the one with the highest
        // confidence and treat it as the same instance (track ID 1) being
        // tracked across frames.
        detection_interfaces::msg::DetectionResult modifiedMsg{*msg};
        if (!msg->instances.empty()) {
          auto obj{*std::max_element(msg->instances.begin(),
                                     msg->instances.end(),
                                     [](const auto& a, const auto& b) {
                                       return a.confidence < b.confidence;
                                     })};
          obj.track_id = 1;
          modifiedMsg.instances = {obj};
        }

        // Update the tracker
        updater.update(modifiedMsg);

        // Attempt to get an active track. If there is already an active track,
        // use it. If there are no active tracks, attempt to activate the next
        // pending track.
        std::optional<std::shared_ptr<const Track>> activeTrackOpt;
        auto activeTracks{tracker->getTracksWithState(Track::State::ACTIVE)};
        if (!activeTracks.empty()) {
          activeTrackOpt = activeTracks[0];
        } else {
          activeTrackOpt = tracker->activateNextPendingTrack();
        }

        if (!activeTrackOpt) {
          return;
        }

        // Push a new lookahead waypoint for the active track
        auto activeTrack{std::move(*activeTrackOpt)};
        double trackLastDetected{activeTrack->getTimestampSecs()};
        double lookaheadTimestampSecs{trackLastDetected + lookaheadSecs};
        Position lookaheadPosition{
            activeTrack->getPredictor().predict(lookaheadTimestampSecs)};
        LaserCoord lookaheadLaserCoord{
            calibration_->cameraPositionToLaserCoord(lookaheadPosition)};
        laser_->addWaypoint(activeTrack->getId(), lookaheadLaserCoord,
                            lookaheadTimestampSecs);
      });

  // Start circle detection
  detection_->startDetection(detection_interfaces::msg::DetectionType::CIRCLE);

  auto nextLaserTime{std::chrono::steady_clock::now()};
  while (!stopSignal) {
    // Strobe the laser on briefly at a fixed interval so the beam stays
    // visible without a continuous burn, for as long as the task is
    // running. Lookahead waypoints are pushed directly from the detections
    // callback above as soon as a new detection comes in.
    float remainingSecs{std::chrono::duration<float>(
                            nextLaserTime - std::chrono::steady_clock::now())
                            .count()};
    if (remainingSecs > 0.0f) {
      // Sleep in short increments so stopSignal is still checked promptly
      std::this_thread::sleep_for(
          std::chrono::duration<float>(std::min(remainingSecs, 0.05f)));
      continue;
    }

    // Strobe the laser on briefly so the beam is visible at the target
    // without staying continuously lit
    laser_->setColor(laserColor);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    laser_->setColor(LaserColor{0.0f, 0.0f, 0.0f, 0.0f});

    nextLaserTime =
        std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<float>(laserIntervalSecs));
  }

  laser_->clearPaths();
  laser_->stop();
  detection_->stopAllDetections();
  detectionCallbackRegistry_->clear();
}
