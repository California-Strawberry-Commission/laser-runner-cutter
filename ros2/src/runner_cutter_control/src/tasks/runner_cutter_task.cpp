#include "runner_cutter_control/tasks/runner_cutter_task.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <thread>
#include <vector>

#include "detection_interfaces/msg/detection_type.hpp"
#include "runner_cutter_control/tasks/detection_tracker_updater.hpp"
#include "runner_cutter_control/tracking/tracker.hpp"

namespace {

std::chrono::system_clock::time_point toTimePoint(double timestampSec) {
  return std::chrono::system_clock::time_point{
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::duration<double>(timestampSec))};
}

}  // namespace

RunnerCutterTask::RunnerCutterTask(
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

void RunnerCutterTask::run(float trackMissTimeoutSecs, int targetAttempts,
                           float lookaheadSecs,
                           const LaserColor& burnLaserColor, float burnTimeSecs,
                           std::atomic<bool>& stopSignal) {
  auto tracker{std::make_shared<Tracker>()};
  DetectionTrackerUpdater updater{tracker, trackMissTimeoutSecs,
                                  targetAttempts};

  // Clear and arm laser
  laser_->clearPaths();
  laser_->setColor(burnLaserColor);
  laser_->play();

  // The track currently being targeted/burned by this task (0 if none)
  uint32_t activeTrackId{0};
  // The time at which burn should end for the active track
  std::optional<std::chrono::system_clock::time_point> burnEndTime;

  // Register to receive per-frame detection updates during the task
  detectionCallbackRegistry_->set(
      [this, lookaheadSecs, burnTimeSecs, tracker, &updater, &stopSignal,
       &activeTrackId, &burnEndTime](
          detection_interfaces::msg::DetectionResult::SharedPtr msg) {
        // Only process runner detections
        if (stopSignal ||
            msg->detection_type !=
                detection_interfaces::msg::DetectionType::RUNNER) {
          return;
        }

        // Update the tracker
        updater.update(*msg);

        // If the active track has been burned for long enough, mark it as
        // completed
        if (activeTrackId != 0 && burnEndTime &&
            std::chrono::system_clock::now() >= *burnEndTime) {
          tracker->transitionTrackState(activeTrackId, Track::State::COMPLETED);
          RCLCPP_INFO(logger_,
                      "Burned track %u for %.2f secs. Marking as COMPLETED.",
                      activeTrackId, burnTimeSecs);
        }

        // Attempt to get an active track. If there is already an active track,
        // use it. If there are no active tracks, attempt to activate the next
        // pending track.
        std::optional<std::shared_ptr<const Track>> newActiveTrackOpt;
        auto activeTracks{tracker->getTracksWithState(Track::State::ACTIVE)};
        if (!activeTracks.empty()) {
          newActiveTrackOpt = activeTracks[0];
        } else {
          newActiveTrackOpt = tracker->activateNextPendingTrack();
        }
        uint32_t newActiveTrackId{
            newActiveTrackOpt ? (*newActiveTrackOpt)->getId() : 0};

        std::vector<LaserControlClient::Waypoint> waypoints;
        std::vector<LaserControlClient::PathState> pathStates;

        // If the active track just changed, disable or remove the previously
        // active track's path.
        if (newActiveTrackId != activeTrackId) {
          if (activeTrackId != 0) {
            auto prevTrackOpt{tracker->getTrack(activeTrackId)};
            bool pending{prevTrackOpt &&
                         (*prevTrackOpt)->getState() == Track::State::PENDING};
            pathStates.push_back(
                {activeTrackId, pending
                                    ? LaserControlClient::PathStatus::DISABLED
                                    : LaserControlClient::PathStatus::REMOVED});
          }
          activeTrackId = newActiveTrackId;
          burnEndTime.reset();
        }

        double lookaheadTimestampSecs{rclcpp::Time(msg->timestamp).seconds() +
                                      lookaheadSecs};
        // Predicts a lookahead waypoint for the given track
        auto predictWaypoint{[this, lookaheadTimestampSecs](
                                 const std::shared_ptr<const Track>& track) {
          Position lookaheadPosition{
              track->getPredictor().predict(lookaheadTimestampSecs)};
          LaserCoord lookaheadLaserCoord{
              calibration_->cameraPositionToLaserCoord(lookaheadPosition)};
          return LaserControlClient::Waypoint{
              track->getId(), lookaheadLaserCoord, lookaheadTimestampSecs};
        }};

        if (newActiveTrackId != 0) {
          auto activeTrack{std::move(*newActiveTrackOpt)};

          if (!burnEndTime) {
            burnEndTime = toTimePoint(lookaheadTimestampSecs + burnTimeSecs);
          }

          // Push a new lookahead waypoint for the active track
          waypoints.push_back(predictWaypoint(activeTrack));
          pathStates.push_back(
              {newActiveTrackId, LaserControlClient::PathStatus::ACTIVE});
        }

        // Also push lookahead waypoints for all pending tracks, so their
        // laser paths stay up to date even though they aren't actively
        // rendered until they become the active track.
        for (const auto& pendingTrack :
             tracker->getTracksWithState(Track::State::PENDING)) {
          waypoints.push_back(predictWaypoint(pendingTrack));
        }

        if (!waypoints.empty() || !pathStates.empty()) {
          laser_->updatePaths(waypoints, pathStates);
        }
      });

  // Start runner detection
  detection_->startDetection(detection_interfaces::msg::DetectionType::RUNNER);

  // Keep the task alive until stopped
  while (!stopSignal) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  laser_->clearPaths();
  laser_->stop();
  detection_->stopAllDetections();
  detectionCallbackRegistry_->clear();
}
