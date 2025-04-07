#include "runner_cutter_control_cpp/prediction/kalman_filter_predictor.hpp"

KalmanFilterPredictor::KalmanFilterPredictor() { reset(); }

void KalmanFilterPredictor::add(std::tuple<float, float, float> position,
                                double timestampMs, float confidence) {
  if (!initialized_) {
    x_.head<3>() = Eigen::Vector3d(std::get<0>(position), std::get<1>(position),
                                   std::get<2>(position));
    initialized_ = true;
  } else {
    double dt{timestampMs - lastTimestampMs_};
    if (dt <= 0.0) {
      // Ignore out-of-order measurements
      return;
    };

    // Update F with dt
    F_(0, 3) = F_(1, 4) = F_(2, 5) = dt;

    // Predict step
    x_ = F_ * x_;                        // x = Fx (+ Bu)
    P_ = F_ * P_ * F_.transpose() + Q_;  // P = FPF' + Q

    // Update step
    R_ = lerpR(confidence, 10.0f, 50.0f);
    Eigen::Vector3d z{std::get<0>(position), std::get<1>(position),
                      std::get<2>(position)};
    Eigen::VectorXd y{z - H_ * x_};
    Eigen::MatrixXd S{H_ * P_ * H_.transpose() + R_};
    Eigen::MatrixXd K{P_ * H_.transpose() * S.inverse()};
    x_ = x_ + K * y;
    P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H_) * P_;
  }

  lastTimestampMs_ = timestampMs;
}

std::tuple<float, float, float> KalmanFilterPredictor::predict(
    double timestampMs) {
  double dt{timestampMs - lastTimestampMs_};
  if (dt <= 0.0) {
    return {x_[0], x_[1], x_[2]};
  }

  Eigen::MatrixXd F_future{Eigen::MatrixXd::Identity(6, 6)};
  // (x', y', z') = (x + vx * dt, y + vy * dt, z + vz * dt)
  F_future(0, 3) = F_future(1, 4) = F_future(2, 5) = dt;
  Eigen::VectorXd x_future{F_future * x_};
  return {x_future[0], x_future[1], x_future[2]};
}

void KalmanFilterPredictor::reset() {
  // Initialize Kalman filter with 6D state (x, y, z, vx, vy, vz)

  // State transition matrix (F) - will be updated later with dt
  F_ = Eigen::MatrixXd::Identity(6, 6);

  // Measurement function (H) - extracts (x, y, z) from state
  H_ = Eigen::MatrixXd::Zero(3, 6);
  H_(0, 0) = 1;
  H_(1, 1) = 1;
  H_(2, 2) = 1;

  // State covariance matrix (P) - initial confidence in state
  P_ = Eigen::MatrixXd::Identity(6, 6) * 1000;

  // Measurement noise covariance matrix (R) - uncertainty in (x, y, z)
  // detection
  // Lower = trust provided measurements more
  // Will be dynamically set later based on confidence of each measurement
  R_ = Eigen::MatrixXd::Identity(3, 3) * 10.0;

  // Process noise covariance matrix (Q) - how much we expect motion to vary
  // over time
  // Lower = smoother, more stable, but slower response to movement changes
  Q_ = Eigen::MatrixXd::Identity(6, 6) * 1e-5;

  // Initial state [x, y, z, vx, vy, vz]
  x_ = Eigen::VectorXd::Zero(6);

  lastTimestampMs_ = 0.0f;
  initialized_ = false;
}

Eigen::MatrixXd KalmanFilterPredictor::lerpR(float confidence, float min,
                                             float max) {
  Eigen::MatrixXd R_min{Eigen::MatrixXd::Identity(3, 3) * min};
  Eigen::MatrixXd R_max{Eigen::MatrixXd::Identity(3, 3) * max};
  return R_max - confidence * (R_max - R_min);
}