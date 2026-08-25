#include "runner_cutter_control/prediction/kalman_filter_predictor.hpp"

KalmanFilterPredictor::KalmanFilterPredictor(
    double measurementNoiseStdMin, double measurementNoiseStdMax,
    double accelerationNoiseStd, double initialPositionStd,
    double initialVelocityStd, double velocityMeasurementNoiseStdMin,
    double velocityMeasurementNoiseStdMax)
    : measurementNoiseStdMin_{measurementNoiseStdMin},
      measurementNoiseStdMax_{measurementNoiseStdMax},
      accelerationNoiseStd_{accelerationNoiseStd},
      initialPositionStd_{initialPositionStd},
      initialVelocityStd_{initialVelocityStd},
      velocityMeasurementNoiseStdMin_{velocityMeasurementNoiseStdMin},
      velocityMeasurementNoiseStdMax_{velocityMeasurementNoiseStdMax} {
  reset();
}

bool KalmanFilterPredictor::add(double timestampSec, const Position& position,
                                float confidence) {
  double dt{timestampSec - getLastTimestampSec()};
  if (!Predictor::add(timestampSec, position, confidence)) {
    return false;
  }

  if (!initialized_) {
    x_.head<3>() = Eigen::Vector3d(position.x, position.y, position.z);
    initialized_ = true;
  } else {
    predictStep(dt);
    Eigen::Vector3d z{position.x, position.y, position.z};
    correctStep(z, H_, measurementNoiseStdMin_, measurementNoiseStdMax_,
                confidence);
  }

  return true;
}

bool KalmanFilterPredictor::addVelocity(double timestampSec,
                                        const Velocity& velocity,
                                        float confidence) {
  double dt{timestampSec - getLastTimestampSec()};
  if (!initialized_ || dt <= 0.0) {
    return false;
  }

  predictStep(dt);
  Eigen::Vector3d zVel{velocity.vx, velocity.vy, velocity.vz};
  correctStep(zVel, Hvel_, velocityMeasurementNoiseStdMin_,
              velocityMeasurementNoiseStdMax_, confidence);
  lastTimestampSec_ = timestampSec;

  return true;
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

  Hvel_.setZero();
  Hvel_(0, 3) = Hvel_(1, 4) = Hvel_(2, 5) = 1.0;

  // Initial uncertainties in state
  P_.setZero();
  P_.block<3, 3>(0, 0) =
      Eigen::Matrix3d::Identity() * (initialPositionStd_ * initialPositionStd_);
  P_.block<3, 3>(3, 3) =
      Eigen::Matrix3d::Identity() * (initialVelocityStd_ * initialVelocityStd_);

  // Will be computed every predictStep
  Q_.setZero();

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

void KalmanFilterPredictor::correctStep(const Eigen::Vector3d& z,
                                        const Matrix3x6d& H, double noiseStdMin,
                                        double noiseStdMax, float confidence) {
  // Measurement noise covariance (R): uncertainty in the measurement.
  // Lower = trust the measurement more. Higher = trust the model more.
  // Adjust based on confidence.
  confidence = std::clamp(confidence, 0.0f, 1.0f);
  double sigma{noiseStdMin + (1.0 - confidence) * (noiseStdMax - noiseStdMin)};
  Eigen::Matrix3d R{Eigen::Matrix3d::Identity() * (sigma * sigma)};

  Eigen::Vector3d y{z - H * x_};                  // innovation
  Eigen::Matrix3d S{H * P_ * H.transpose() + R};  // innovation covariance
  Eigen::Matrix<double, 6, 3> K{P_ * H.transpose() *
                                S.inverse()};  // Kalman gain

  // Update state
  x_ = x_ + K * y;

  // Update covariance: P = P - K * H * P;
  // Use the Joseph form to protect against problems with roundoff error:
  // A = I - K * H
  // P = A * P * A' + K * R * K'
  Matrix6d A{Matrix6d::Identity() - K * H};
  P_ = A * P_ * A.transpose() + K * R * K.transpose();
}
