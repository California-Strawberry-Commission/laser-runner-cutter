#include "runner_cutter_control/prediction/kalman_filter_predictor.hpp"

KalmanFilterPredictor::KalmanFilterPredictor() { reset(); }

void KalmanFilterPredictor::add(double timestampSec,
                                const Measurement& measurement) {
  double dt{timestampSec - getLastTimestampSec()};
  Predictor::add(timestampSec, measurement);

  if (!initialized_) {
    x_.head<3>() = Eigen::Vector3d(
        measurement.position.x, measurement.position.y, measurement.position.z);
    initialized_ = true;
  } else {
    if (dt <= 0.0) {
      // Ignore out-of-order or duplicate timestamp
      return;
    };

    predictStep(dt);
    Eigen::Vector3d z{measurement.position.x, measurement.position.y,
                      measurement.position.z};
    updateStep(z, measurement.confidence);
  }
}

Position KalmanFilterPredictor::predict(double timestampSec) const {
  double dt{timestampSec - getLastTimestampSec()};

  if (dt <= 0.0) {
    return interpolated(timestampSec);
  }

  Eigen::MatrixXd F_future{Eigen::MatrixXd::Identity(6, 6)};
  // (x', y', z') = (x + vx * dt, y + vy * dt, z + vz * dt)
  F_future(0, 3) = F_future(1, 4) = F_future(2, 5) = dt;
  Eigen::VectorXd x_future{F_future * x_};
  return {static_cast<float>(x_future[0]), static_cast<float>(x_future[1]),
          static_cast<float>(x_future[2])};
}

void KalmanFilterPredictor::reset() {
  Predictor::reset();

  // Initial state [x, y, z, vx, vy, vz]
  x_.setZero();

  // Will be updated later with dt
  F_ = Matrix6d::Identity();

  H_.setZero();
  H_(0, 0) = 1.0;
  H_(1, 1) = 1.0;
  H_(2, 2) = 1.0;

  // Large initial uncertainty in state
  P_ = Matrix6d::Identity() * 1000;

  Q_ = Matrix6d::Identity(6, 6) * 1e-5;

  // Will be dynamically set later based on confidence of each measurement
  R_ = Eigen::Matrix3d::Identity() * 10.0;

  initialized_ = false;
}

void KalmanFilterPredictor::predictStep(double dt) {
  // Update F with dt
  F_(0, 3) = F_(1, 4) = F_(2, 5) = dt;

  x_ = F_ * x_;                        // x = Fx (+ Bu)
  P_ = F_ * P_ * F_.transpose() + Q_;  // P = FPF' + Q
}

void KalmanFilterPredictor::updateStep(const Eigen::Vector3d& z,
                                       float confidence) {
  R_ = lerpR(std::clamp(confidence, 0.01f, 1.0f), 10.0, 50.0);
  Eigen::Vector3d y{z - H_ * x_};                    // innovation
  Eigen::Matrix3d S{H_ * P_ * H_.transpose() + R_};  // innovation covariance
  Eigen::Matrix<double, 6, 3> K{P_ * H_.transpose() *
                                S.inverse()};  // Kalman gain

  // Update state
  x_ = x_ + K * y;

  // Update covariance
  P_ = (Matrix6d::Identity() - K * H_) * P_;
}

Eigen::Matrix3d KalmanFilterPredictor::lerpR(float confidence, double min,
                                             double max) {
  Eigen::Matrix3d R_min{Eigen::Matrix3d::Identity() * min};
  Eigen::Matrix3d R_max{Eigen::Matrix3d::Identity() * max};
  return R_max - confidence * (R_max - R_min);
}
