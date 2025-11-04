#include "runner_cutter_control/clients/laser_detection_context.hpp"

LaserDetectionContext::LaserDetectionContext(
    std::shared_ptr<LaserControlClient> laser,
    std::shared_ptr<CameraControlClient> camera)
    : laser_{std::move(laser)}, camera_{std::move(camera)}, restored_{false} {
  prevExposureUs_ = camera_->getExposure();
  prevGainDb_ = camera_->getGain();

  laser_->clearPoint();
  laser_->play();
  camera_->setExposure(1.0f);
  camera_->setGain(0.0f);
}

LaserDetectionContext::~LaserDetectionContext() { restore(); }

void LaserDetectionContext::restore() {
  if (restored_) {
    return;
  }

  laser_->clearPoint();
  laser_->stop();
  camera_->setGain(prevGainDb_);
  camera_->setExposure(prevExposureUs_);
  restored_ = true;
}