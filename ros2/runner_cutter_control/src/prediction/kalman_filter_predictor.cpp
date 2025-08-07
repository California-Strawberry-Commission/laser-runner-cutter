#include "runner_cutter_control/prediction/kalman_filter_predictor.hpp"

KalmanFilterPredictor::KalmanFilterPredictor() { reset(); }

void KalmanFilterPredictor::add(double timestampMs,
                                const Measurement& measurement) {
  double dt{timestampMs - getLastTimestampMs()};
  Predictor::add(timestampMs, measurement);

  if (!initialized_) {
    x_.head<3>() = Eigen::Vector3d(
        measurement.position.x, measurement.position.y, measurement.position.z);
    initialized_ = true;
  } else {
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
    R_ = lerpR(measurement.confidence, 10.0f, 50.0f);
    Eigen::Vector3d z{measurement.position.x, measurement.position.y,
                      measurement.position.z};
    Eigen::VectorXd y{z - H_ * x_};
    Eigen::MatrixXd S{H_ * P_ * H_.transpose() + R_};
    Eigen::MatrixXd K{P_ * H_.transpose() * S.inverse()};
    x_ = x_ + K * y;
    P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H_) * P_;
  }
}

Position KalmanFilterPredictor::predict(double timestampMs) const {
  double dt{timestampMs - getLastTimestampMs()};

  if (dt <= 0.0) {
    return interpolated(timestampMs);
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

  initialized_ = false;
}

Eigen::MatrixXd KalmanFilterPredictor::lerpR(float confidence, float min,
                                             float max) {
  Eigen::MatrixXd R_min{Eigen::MatrixXd::Identity(3, 3) * min};
  Eigen::MatrixXd R_max{Eigen::MatrixXd::Identity(3, 3) * max};
  return R_max - confidence * (R_max - R_min);
}
