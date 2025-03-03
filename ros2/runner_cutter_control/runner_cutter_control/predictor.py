from abc import ABC, abstractmethod
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter


class Predictor(ABC):
    @abstractmethod
    def add(
        self,
        position: Tuple[float, float, float],
        timestamp_ms: float,
        confidence: float = 1.0,
    ):
        """
        Add a new position measurement to the predictor. Calls to `add` for measurements must be
        done in sequential order with respect to their timestamps.

        Args:
            position (Tuple[float, float, float]): Position measurement (x, y, z).
            timestamp_ms (float): Timestamp (in ms) associated with the measurement.
            confidence (float): Confidence score associated with the measurement.
        """
        pass

    @abstractmethod
    def predict(self, timestamp_ms: float) -> Tuple[float, float, float]:
        """
        Predict a future position.

        Args:
            timestamp_ms (float): Timestamp (in ms) to predict the measurement for.
        """
        pass

    @abstractmethod
    def reset(self):
        pass


class KalmanFilterPredictor:
    def __init__(self):
        self._kf = self._create_kalman_filter()
        self._last_timestamp_ms = 0.0

    def add(
        self,
        position: Tuple[float, float, float],
        timestamp_ms: float,
        confidence: float = 1.0,
    ):
        """
        Add a new position measurement to the predictor. Calls to `add` for measurements must be
        done in sequential order with respect to their timestamps.

        Args:
            position (Tuple[float, float, float]): Position measurement (x, y, z).
            timestamp_ms (float): Timestamp (in ms) associated with the measurement.
            confidence (float): Confidence score associated with the measurement.
        """
        if self._last_timestamp_ms == 0.0:
            # If this is the first measurement, set the initial state vector
            self._kf.x = np.array([position[0], position[1], position[2], 0, 0, 0])
        else:
            dt = timestamp_ms - self._last_timestamp_ms
            if dt < 0:
                # Ignore out-of-order measurements
                return

            # Update transition matrix dynamically
            self._kf.F[0, 3] = dt  # update dt in F matrix (x)
            self._kf.F[1, 4] = dt  # update dt in F matrix (y)
            self._kf.F[2, 5] = dt  # update dt in F matrix (z)

            # Update measurement noise matrix dynamically based on confidence
            # Lower R when confidence is high, and higher R when confidence is low
            R_min = np.eye(3) * 10
            R_max = np.eye(3) * 50
            self._kf.R = R_max - (confidence * (R_max - R_min))

            self._kf.predict()
            self._kf.update(np.array(position))

        self._last_timestamp_ms = timestamp_ms

    def predict(self, timestamp_ms: float) -> Tuple[float, float, float]:
        """
        Predict a future position.

        Args:
            timestamp_ms (float): Timestamp (in ms) to predict the measurement for.
        """
        dt = timestamp_ms - self._last_timestamp_ms
        if dt <= 0:
            # If time is in the past, return current estimate
            return self._kf.x[:3]

        F_future = np.array(
            [
                [1, 0, 0, dt, 0, 0],  # x = x + vx * dt
                [0, 1, 0, 0, dt, 0],  # y = y + vy * dt
                [0, 0, 1, 0, 0, dt],  # z = z + vz * dt
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        predicted_state = F_future @ self._kf.x
        return predicted_state[:3]

    def reset(self):
        self._kf = self._create_kalman_filter()
        self._last_timestamp_ms = 0.0

    def _create_kalman_filter(self) -> KalmanFilter:
        # Initialize Kalman filter with 6D state (x, y, z, vx, vy, vz)
        kf = KalmanFilter(dim_x=6, dim_z=3)
        # State transition matrix (F) - will be updated later with dt
        kf.F = np.array(
            [
                [1, 0, 0, 1, 0, 0],  # x = x + vx * dt
                [0, 1, 0, 0, 1, 0],  # y = y + vy * dt
                [0, 0, 1, 0, 0, 1],  # z = z + vz * dt
                [0, 0, 0, 1, 0, 0],  # vx remains the same
                [0, 0, 0, 0, 1, 0],  # vy remains the same
                [0, 0, 0, 0, 0, 1],  # vz remains the same
            ]
        )
        # Measurement function (H) - extracts (x, y, z) from state
        kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        # State covariance matrix (P) - initial confidence in state
        kf.P = np.eye(6) * 1000
        # Measurement noise covariance matrix (R) - uncertainty in (x, y, z) detection
        # Lower = trust provided measurements more
        # Will be dynamically set later based on confidence of each measurement
        kf.R = np.eye(3) * 10
        # Process noise covariance matrix (Q) - how much we expect motion to vary over time
        # Lower = smoother, more stable, but slower response to movement changes
        kf.Q = np.eye(6) * 0.00001
        # Initial state [x, y, z, vx, vy, vz]
        kf.x = np.array([0, 0, 0, 0, 0, 0])
        return kf


if __name__ == "__main__":
    predictor = KalmanFilterPredictor()

    # Sample data
    data = [  # moving
        (1740157598942.124, (41.0, -64.5, 432.75)),
        (1740157599001.046, (42.25, -70.75, 446.25)),
        (1740157599059.6873, (42.0, -74.75, 434.0)),
        (1740157599118.7654, (43.25, -81.5, 447.5)),
        (1740157599177.7407, (43.0, -85.25, 444.5)),
        (1740157599236.8506, (42.75, -88.5, 442.75)),
        (1740157599297.516, (42.5, -94.25, 437.25)),
        (1740157599356.225, (40.5, -94.5, 426.75)),
        (1740157599414.5542, (40.5, -97.75, 426.5)),
    ]

    """
    data = [  # static
        (1740155698317.263, (51.75, -75.5, 427.0)),
        (1740155698376.7227, (52.25, -77.5, 431.5)),
        (1740155698434.172, (51.5, -75.5, 426.5)),
        (1740155698493.7585, (51.75, -75.75, 428.5)),
        (1740155698551.7332, (52.0, -76.5, 430.25)),
        (1740155698610.8508, (51.5, -75.5, 426.25)),
        (1740155698671.3154, (51.75, -76.5, 427.0)),
        (1740155698729.7744, (51.75, -76.5, 428.0)),
        (1740155698789.1252, (52.0, -76.5, 430.25)),
    ]
    """

    true_positions = []
    predicted_positions = []

    # Run Kalman filter
    for timestamp_ms, position in data:
        predictor.add(position, timestamp_ms)

        true_positions.append(position)

    true_positions = np.array(true_positions)

    time_deltas = [data[i + 1][0] - data[i][0] for i in range(len(data) - 1)]
    average_time_delta = sum(time_deltas) / len(time_deltas)
    last_data_timestamp = data[-1][0]
    for i in range(5):
        future_timestamp_ms = last_data_timestamp + average_time_delta * (i + 1)
        predicted_position = predictor.predict(future_timestamp_ms)
        print(f"predicted_position: {predicted_position} @ {future_timestamp_ms}")
        predicted_positions.append(predicted_position)
    predicted_positions = np.array(predicted_positions)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(
        true_positions[:, 0], true_positions[:, 1], "ro", label="Noisy Measurements"
    )
    plt.plot(
        predicted_positions[:, 0],
        predicted_positions[:, 1],
        "bo",
        label="Kalman Filter Prediction",
    )
    plt.legend()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(0, 100)
    plt.ylim(-150, -50)
    plt.title("2D Kalman Filter Tracking")
    plt.grid()
    plt.show()
