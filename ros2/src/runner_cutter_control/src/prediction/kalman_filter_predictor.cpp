#include "runner_cutter_control/prediction/kalman_filter_predictor.hpp"

KalmanFilterPredictor::KalmanFilterPredictor(double measurementNoiseStdMin,
                                             double measurementNoiseStdMax,
                                             double accelerationNoiseStd,
                                             double initialPositionStd,
                                             double initialVelocityStd)
    : measurementNoiseStdMin_{measurementNoiseStdMin},
      measurementNoiseStdMax_{measurementNoiseStdMax},
      accelerationNoiseStd_{accelerationNoiseStd},
      initialPositionStd_{initialPositionStd},
      initialVelocityStd_{initialVelocityStd} {
  reset();
}

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

  Matrix6d F_future{Matrix6d::Identity()};
  // (x', y', z') = (x + vx * dt, y + vy * dt, z + vz * dt)
  F_future(0, 3) = F_future(1, 4) = F_future(2, 5) = dt;
  Vector6d x_future{F_future * x_};
  return {static_cast<float>(x_future[0]), static_cast<float>(x_future[1]),
          static_cast<float>(x_future[2])};
}

void KalmanFilterPredictor::reset() {
  Predictor::reset();

  // Constant-velocity model - initial state [x, y, z, vx, vy, vz]
  x_.setZero();

  // Will be updated every predictStep with dt
  F_.setIdentity();

  H_.setZero();
  H_(0, 0) = H_(1, 1) = H_(2, 2) = 1.0;

  // Initial uncertainties in state
  P_.setZero();
  P_.block<3, 3>(0, 0) =
      Eigen::Matrix3d::Identity() * (initialPositionStd_ * initialPositionStd_);
  P_.block<3, 3>(3, 3) =
      Eigen::Matrix3d::Identity() * (initialVelocityStd_ * initialVelocityStd_);

  // Will be computed every predictStep
  Q_.setZero();

  // Will be dynamically set later based on confidence of each measurement
  R_ = Eigen::Matrix3d::Identity() *
       (measurementNoiseStdMin_ * measurementNoiseStdMin_);

  initialized_ = false;
}

void KalmanFilterPredictor::predictStep(double dt) {
  // Update F with dt
  F_(0, 3) = F_(1, 4) = F_(2, 5) = dt;

  // Update Q - scale process noise by time step
  // Use standard constant velocity model with acceleration noise
  Q_.setZero();
  double dt2{dt * dt};
  double dt3{dt2 * dt};
  double dt4{dt3 * dt};
  double q{accelerationNoiseStd_ * accelerationNoiseStd_};
  // top-left: position-position
  Q_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (dt4 / 4.0) * q;
  // top-right & bottom-left: position-velocity
  Q_.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * (dt3 / 2.0) * q;
  Q_.block<3, 3>(3, 0) = Q_.block<3, 3>(0, 3);
  // bottom-right: velocity-velocity
  Q_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * dt2 * q;

  // Predict state forward: x = Fx (+ Bu)
  x_ = F_ * x_;

  // Predict covariance forward: P = F * P * F' + Q
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilterPredictor::updateStep(const Eigen::Vector3d& z,
                                       float confidence) {
  // Adjust measurement noise covariance based on confidence
  confidence = std::clamp(confidence, 0.0f, 1.0f);
  double sigma{measurementNoiseStdMin_ +
               (1.0 - confidence) *
                   (measurementNoiseStdMax_ - measurementNoiseStdMin_)};
  R_ = Eigen::Matrix3d::Identity() * (sigma * sigma);

  Eigen::Vector3d y{z - H_ * x_};                    // innovation
  Eigen::Matrix3d S{H_ * P_ * H_.transpose() + R_};  // innovation covariance
  Eigen::Matrix<double, 6, 3> K{P_ * H_.transpose() *
                                S.inverse()};  // Kalman gain

  // Update state
  x_ = x_ + K * y;

  // Update covariance: P = P - K * H * P;
  // Use the Joseph form to protect against problems with roundoff error:
  // A = I - K * H
  // P = A * P * A' + K * R * K'
  Matrix6d A{Matrix6d::Identity() - K * H_};
  P_ = A * P_ * A.transpose() + K * R_ * K.transpose();
}
