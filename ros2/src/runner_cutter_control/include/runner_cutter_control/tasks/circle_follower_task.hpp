#pragma once

#include <atomic>

#include "common/event.hpp"
#include "rclcpp/rclcpp.hpp"
#include "runner_cutter_control/calibration/calibration.hpp"
#include "runner_cutter_control/clients/detection_client.hpp"
#include "runner_cutter_control/clients/laser_control_client.hpp"
#include "runner_cutter_control/common_types.hpp"
#include "runner_cutter_control/tracking/tracker.hpp"

/**
 * Test/demo task that continuously follows a detected circle target: unlike
 * LaserTargeting's static aim/burn, this predicts the target's future
 * position and adds laser waypoints ahead of it, strobing the laser on
 * briefly at a fixed interval so the beam stays visible without a
 * continuous burn.
 */
class CircleFollowerTask {
 public:
  CircleFollowerTask(std::shared_ptr<LaserControlClient> laser,
                     std::shared_ptr<DetectionClient> detection,
                     std::shared_ptr<Calibration> calibration,
                     std::shared_ptr<Tracker> tracker, rclcpp::Logger logger);
  ~CircleFollowerTask() = default;

  void run(float laserIntervalSecs, const LaserColor& trackingLaserColor,
           std::atomic<bool>& stopSignal,
           common::Event& pendingTracksChangedEvent,
           common::Event& trackUpdatedEvent);

 private:
  std::shared_ptr<LaserControlClient> laser_;
  std::shared_ptr<DetectionClient> detection_;
  std::shared_ptr<Calibration> calibration_;
  std::shared_ptr<Tracker> tracker_;
  rclcpp::Logger logger_;
};
