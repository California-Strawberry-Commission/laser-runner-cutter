#pragma once

#include "runner_cutter_control/clients/camera_control_client.hpp"
#include "runner_cutter_control/clients/laser_control_client.hpp"

/**
 * Manages the state of laser and camera settings during laser detection.
 *
 * This class ensures that any modified camera settings (such as exposure and
 * gain) are restored when `restore()` is explicitly called, or the instance of
 * LaserDetectionContext is destroyed.
 */
class LaserDetectionContext {
 public:
  explicit LaserDetectionContext(std::shared_ptr<LaserControlClient> laser,
                                 std::shared_ptr<CameraControlClient> camera);
  ~LaserDetectionContext();

  void restore();

 private:
  std::shared_ptr<LaserControlClient> laser_;
  std::shared_ptr<CameraControlClient> camera_;
  float prevExposureUs_{-1.0f};
  float prevGainDb_{-1.0f};
  bool restored_{false};
};