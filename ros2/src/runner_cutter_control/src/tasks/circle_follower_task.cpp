#include "runner_cutter_control/tasks/circle_follower_task.hpp"

#include <chrono>
#include <thread>

#include "detection_interfaces/msg/detection_type.hpp"

CircleFollowerTask::CircleFollowerTask(
    std::shared_ptr<LaserControlClient> laser,
    std::shared_ptr<DetectionClient> detection,
    std::shared_ptr<Calibration> calibration, std::shared_ptr<Tracker> tracker,
    rclcpp::Logger logger)
    : laser_(std::move(laser)),
      detection_(std::move(detection)),
      calibration_(std::move(calibration)),
      tracker_(std::move(tracker)),
      logger_(std::move(logger)) {}

void CircleFollowerTask::run(float laserIntervalSecs,
                             const LaserColor& trackingLaserColor,
                             std::atomic<bool>& stopSignal,
                             common::Event& pendingTracksChangedEvent,
                             common::Event& trackUpdatedEvent) {
  laser_->clearPaths();
  laser_->setColor(LaserColor{0.0f, 0.0f, 0.0f, 0.0f});
  // Arm laser and start detection
  laser_->play();
  detection_->startDetection(detection_interfaces::msg::DetectionType::CIRCLE);

  while (!stopSignal) {
    auto trackOpt{tracker_->activateNextPendingTrack()};
    if (!trackOpt) {
      pendingTracksChangedEvent.wait();
      pendingTracksChangedEvent.clear();
      continue;
    }

    auto track{std::move(*trackOpt)};
    RCLCPP_INFO(logger_, "Following circle");

    // Continuously follow the track for as long as the task is running. A
    // new waypoint is added as soon as a new detection comes in for the
    // track. The laser is strobed every laserIntervalSecs.
    auto nextLaserTime{std::chrono::steady_clock::now()};
    while (!stopSignal) {
      float remainingSecs{std::chrono::duration<float>(
                              nextLaserTime - std::chrono::steady_clock::now())
                              .count()};
      if (remainingSecs > 0.0f && trackUpdatedEvent.wait_for(remainingSecs)) {
        trackUpdatedEvent.clear();
        if (stopSignal) {
          break;
        }

        constexpr double LOOKAHEAD_SECS{0.2};
        double trackLastDetected{track->getTimestampSecs()};
        double lookaheadTimestampSecs{trackLastDetected + LOOKAHEAD_SECS};
        Position lookaheadPosition{
            track->getPredictor().predict(lookaheadTimestampSecs)};
        LaserCoord lookaheadLaserCoord{
            calibration_->cameraPositionToLaserCoord(lookaheadPosition)};

        laser_->addWaypoint(track->getId(), lookaheadLaserCoord,
                            lookaheadTimestampSecs);
        continue;
      }

      if (stopSignal) {
        break;
      }

      // Strobe the laser on briefly so the beam is visible at the target
      // without staying continuously lit
      laser_->setColor(trackingLaserColor);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      laser_->setColor(LaserColor{0.0f, 0.0f, 0.0f, 0.0f});

      nextLaserTime =
          std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<float>(laserIntervalSecs));
    }
  }

  laser_->clearPaths();
  laser_->stop();
  detection_->stopAllDetections();
}
