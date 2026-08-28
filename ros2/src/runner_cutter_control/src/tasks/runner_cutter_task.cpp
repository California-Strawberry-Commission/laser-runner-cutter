#include "runner_cutter_control/tasks/runner_cutter_task.hpp"

#include <chrono>
#include <cstdint>
#include <thread>
#include <unordered_map>

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

  // Records the time the laser should stop burning each active track, so its
  // path can be removed once it has been burned for burnTimeSecs.
  std::unordered_map<uint32_t, std::chrono::system_clock::time_point>
      burnEndTimes;

  // Register to receive per-frame detection updates during the task
  detectionCallbackRegistry_->set(
      [this, lookaheadSecs, burnTimeSecs, tracker, &updater, &stopSignal,
       &burnEndTimes](
          detection_interfaces::msg::DetectionResult::SharedPtr msg) {
        // Only process runner detections
        if (stopSignal ||
            msg->detection_type !=
                detection_interfaces::msg::DetectionType::RUNNER) {
          return;
        }

        // Update the tracker
        updater.update(*msg);

        // Remove laser paths for tracks we were burning that are no longer
        // active
        for (auto it{burnEndTimes.begin()}; it != burnEndTimes.end();) {
          auto trackOpt{tracker->getTrack(it->first)};
          if (trackOpt && (*trackOpt)->getState() == Track::State::ACTIVE) {
            ++it;
            continue;
          }
          laser_->removePath(it->first);
          it = burnEndTimes.erase(it);
        }

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

        auto activeTrack{std::move(*activeTrackOpt)};
        uint32_t activeTrackId{activeTrack->getId()};
        double lastDetected{activeTrack->getTimestampSecs()};
        double lookaheadTimestampSecs{lastDetected + lookaheadSecs};

        // Set the burn end time the first time this track becomes active: the
        // laser reaches the track's path at lookaheadTimestampSecs and burns it
        // for burnTimeSecs. Once we reach that time, mark it as completed and
        // remove its laser path.
        auto [burnEnd, isNewBurn]{burnEndTimes.try_emplace(
            activeTrackId, toTimePoint(lookaheadTimestampSecs + burnTimeSecs))};
        if (!isNewBurn && std::chrono::system_clock::now() >= burnEnd->second) {
          laser_->removePath(activeTrackId);
          tracker->transitionTrackState(activeTrackId, Track::State::COMPLETED);
          burnEndTimes.erase(burnEnd);
          RCLCPP_INFO(logger_,
                      "Burned track %u for %.2f secs. Marking as COMPLETED.",
                      activeTrackId, burnTimeSecs);
          return;
        }

        // Push a new lookahead waypoint for the active track
        Position lookaheadPosition{
            activeTrack->getPredictor().predict(lookaheadTimestampSecs)};
        LaserCoord lookaheadLaserCoord{
            calibration_->cameraPositionToLaserCoord(lookaheadPosition)};
        laser_->addWaypoint(activeTrackId, lookaheadLaserCoord,
                            lookaheadTimestampSecs);
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
