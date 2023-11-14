import numpy as np
import time
from scipy.optimize import minimize


class Calibration:
    def __init__(self, laser, camera, laser_color, logger=None):
        self.laser = laser
        self.camera = camera
        self.logger = logger
        self.laser_color = laser_color

        self.calibration_laser_pixels = []
        self.calibration_camera_points = []

        self.is_calibrated = False
        self.camera_to_laser_transform = np.zeros((4, 3))

    def calibrate(self):
        """Use image correspondences to compute the transformation matrix from camera to laser

        1. Shoot the laser at predetermined points
        2. For each laser point, capture an image from the camera
        3. Identify the corresponding point in the camera frame
        4. Compute the transformation matrix from camera to laser
        """

        # TODO: Depth values are noisy when the laser is on. Figure out how to reduce noise, or
        # use a depth frame obtained when the laser is off

        # Get calibration points
        x_bounds = [float("inf"), float("-inf")]
        y_bounds = [float("inf"), float("-inf")]
        laser_bounds = self.laser.get_bounds(1.0)
        for x, y in laser_bounds:
            if x < x_bounds[0]:
                x_bounds[0] = x
            if x > x_bounds[1]:
                x_bounds[1] = x
            if y < y_bounds[0]:
                y_bounds[0] = y
            if y > y_bounds[1]:
                y_bounds[1] = y

        grid_size = (5, 5)
        x_step = (x_bounds[1] - x_bounds[0]) / (grid_size[0] - 1)
        y_step = (y_bounds[1] - y_bounds[0]) / (grid_size[1] - 1)
        pending_calibration_laser_pixels = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x = x_bounds[0] + i * x_step
                y = y_bounds[0] + j * y_step
                pending_calibration_laser_pixels.append((x, y))

        # Get image correspondences
        self.calibration_laser_pixels = []
        self.calibration_camera_points = []

        # TODO: set exposure on camera node automatically when detecting laser
        self.camera.set_exposure(0.001)

        for laser_pixel in pending_calibration_laser_pixels:
            self.laser.stop_laser()
            self.laser.start_laser(point=laser_pixel, color=self.laser_color)
            # Wait for galvo to settle and for camera frame capture
            time.sleep(0.2)
            self.add_point_correspondence(laser_pixel)

        self.laser.stop_laser()
        self.camera.set_exposure(-1.0)

        if self.logger:
            self.logger.info(
                f"{len(self.calibration_laser_pixels)} out of {len(pending_calibration_laser_pixels)} point correspondences found."
            )

        if len(self.calibration_laser_pixels) < 3:
            if self.logger:
                self.logger.warning(
                    "Calibration failed: insufficient point correspondences found."
                )
            return False

        # Use least squares for an initial estimate, then use bundle adjustment to refine
        self._update_transform_least_squares()
        self.update_transform_bundle_adjustment()
        self.is_calibrated = True

        return True

    def camera_point_to_laser_pixel(self, camera_point):
        homogeneous_camera_point = np.hstack((camera_point, 1))
        transformed_point = homogeneous_camera_point @ self.camera_to_laser_transform
        transformed_point = transformed_point / transformed_point[2]
        return transformed_point[:2]

    def add_point_correspondence(self, laser_pixel):
        attempts = 0
        while attempts < 50:
            attempts += 1
            pos_data = self.camera.get_laser_pos()
            if pos_data["pos_list"]:
                # can add error if len greater then 1
                self.calibration_laser_pixels.append(laser_pixel)
                self.calibration_camera_points.append(pos_data["pos_list"][0])
                if self.logger:
                    self.logger.info(
                        f"Added point correspondence. Total correspondences = {len(self.calibration_laser_pixels)}"
                    )
                break

            # Should be a ros way to sleep
            time.sleep(0.005)

    def update_transform_bundle_adjustment(self):
        def cost_function(parameters, camera_points, laser_pixels):
            homogeneous_camera_points = np.hstack(
                (camera_points, np.ones((camera_points.shape[0], 1)))
            )
            transform = parameters.reshape((4, 3))
            transformed_points = np.dot(homogeneous_camera_points, transform)
            homogeneous_transformed_points = (
                transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]
            )
            errors = np.sum(
                (homogeneous_transformed_points[:, :2] - laser_pixels) ** 2, axis=1
            )
            return np.mean(errors)

        laser_pixels = np.array(self.calibration_laser_pixels)
        camera_points = np.array(self.calibration_camera_points)
        result = minimize(
            cost_function,
            self.camera_to_laser_transform,
            args=(camera_points, laser_pixels),
            method="L-BFGS-B",
        )
        self.camera_to_laser_transform = result.x.reshape((4, 3))

        if self.logger:
            self.logger.info("Updated camera to laser transform")
            self.logger.info(f"Laser pixels:\n{laser_pixels}")
            homogeneous_camera_points = np.hstack(
                (camera_points, np.ones((camera_points.shape[0], 1)))
            )
            transformed_points = np.dot(
                homogeneous_camera_points, self.camera_to_laser_transform
            )
            homogeneous_transformed_points = (
                transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]
            )
            self.logger.info(
                f"Laser pixels calculated using transform:\n{homogeneous_transformed_points[:, :2]}"
            )

    def _update_transform_least_squares(self):
        laser_pixels = np.array(self.calibration_laser_pixels)
        camera_points = np.array(self.calibration_camera_points)
        homogeneous_laser_pixels = np.hstack(
            (laser_pixels, np.ones((laser_pixels.shape[0], 1)))
        )
        homogeneous_camera_points = np.hstack(
            (camera_points, np.ones((camera_points.shape[0], 1)))
        )
        self.camera_to_laser_transform = np.linalg.lstsq(
            homogeneous_camera_points,
            homogeneous_laser_pixels,
            rcond=None,
        )[0]

        if self.logger:
            self.logger.info("Updated camera to laser transform")
            self.logger.info(f"Laser pixels:\n{laser_pixels}")
            transformed_points = np.dot(
                homogeneous_camera_points, self.camera_to_laser_transform
            )
            homogeneous_transformed_points = (
                transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]
            )
            self.logger.info(
                f"Laser pixels calculated using transform:\n{homogeneous_transformed_points[:, :2]}"
            )
