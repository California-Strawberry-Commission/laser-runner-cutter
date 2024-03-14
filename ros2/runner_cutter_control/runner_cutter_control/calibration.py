import asyncio

import numpy as np
from scipy.optimize import minimize


class Calibration:
    def __init__(self, laser_client, camera_client, laser_color, logger=None):
        self.laser_client = laser_client
        self.camera_client = camera_client
        self.logger = logger
        self.laser_color = laser_color

        self.calibration_laser_pixels = []
        self.calibration_camera_points = []

        self.is_calibrated = False
        self.camera_to_laser_transform = np.zeros((4, 3))

    async def calibrate(self):
        """Use image correspondences to compute the transformation matrix from camera to laser

        1. Shoot the laser at predetermined points
        2. For each laser point, capture an image from the camera
        3. Identify the corresponding point in the camera frame
        4. Compute the transformation matrix from camera to laser
        """

        # TODO: Depth values are noisy when the laser is on. Figure out how to reduce noise, or
        # use a depth frame obtained when the laser is off

        # Get calibration points
        grid_size = (5, 5)
        x_step = 1.0 / (grid_size[0] - 1)
        y_step = 1.0 / (grid_size[1] - 1)
        pending_calibration_laser_pixels = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x = i * x_step
                y = j * y_step
                pending_calibration_laser_pixels.append((x, y))

        # Get image correspondences
        self.calibration_laser_pixels = []
        self.calibration_camera_points = []

        await self.add_calibration_points(pending_calibration_laser_pixels)

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

    async def add_calibration_points(self, laser_pixels, update_transform=False):
        # TODO: set exposure on camera node automatically when detecting laser
        await self.camera_client.set_exposure(0.001)
        await self.laser_client.set_color((0.0, 0.0, 0.0))
        await self.laser_client.start_laser()
        for laser_pixel in laser_pixels:
            await self.laser_client.set_point(laser_pixel)
            await self.laser_client.set_color(self.laser_color)
            # Wait for galvo to settle and for camera frame capture
            # TODO: increase frame capture rate and reduce this time
            await asyncio.sleep(1)
            await self.add_point_correspondence(laser_pixel)
            await self.laser_client.set_color((0.0, 0.0, 0.0))

        await self.laser_client.stop_laser()
        await self.camera_client.set_exposure(-1.0)

        if update_transform:
            self.update_transform_bundle_adjustment()

    def camera_point_to_laser_pixel(self, camera_point):
        homogeneous_camera_point = np.hstack((camera_point, 1))
        transformed_point = homogeneous_camera_point @ self.camera_to_laser_transform
        transformed_point = transformed_point / transformed_point[2]
        return transformed_point[:2]

    async def add_point_correspondence(self, laser_pixel):
        attempts = 0
        while attempts < 20:
            self.logger.info(f"attempt = {attempts}")
            attempts += 1
            pos_data = await self.camera_client.get_laser_pos()
            if pos_data["pos_list"]:
                # can add error if len greater then 1
                self.calibration_laser_pixels.append(laser_pixel)
                self.calibration_camera_points.append(pos_data["pos_list"][0])
                if self.logger:
                    camera_pixel = (
                        pos_data["point_list"][0] if pos_data["point_list"] else "None"
                    )
                    self.logger.info(
                        f"Found point correspondence: laser_pixel = {laser_pixel}, camera_pixel = {camera_pixel}. {len(self.calibration_laser_pixels)} total correspondences."
                    )
                return
            await asyncio.sleep(0.2)
        self.logger.info(
            f"Failed to find point. {len(self.calibration_laser_pixels)} total correspondences."
        )

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
            error = np.mean(
                np.sqrt(
                    np.sum(
                        (homogeneous_transformed_points[:, :2] - laser_pixels) ** 2,
                        axis=1,
                    )
                )
            )
            self.logger.info(f"Mean reprojection error: {error}")

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
            error = np.mean(
                np.sqrt(
                    np.sum(
                        (homogeneous_transformed_points[:, :2] - laser_pixels) ** 2,
                        axis=1,
                    )
                )
            )
            self.logger.info(f"Mean reprojection error: {error}")
