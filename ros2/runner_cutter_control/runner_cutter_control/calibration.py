import asyncio
import logging

import numpy as np
from scipy.optimize import minimize


class Calibration:
    def __init__(self, laser_client, camera_client, laser_color, logger=None):
        self.laser_client = laser_client
        self.camera_client = camera_client
        self.laser_color = laser_color
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)

        self.calibration_laser_coords = []
        self.calibration_camera_positions = []
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
        pending_calibration_laser_coords = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x = i * x_step
                y = j * y_step
                pending_calibration_laser_coords.append((x, y))

        # Get image correspondences
        self.calibration_laser_coords = []
        self.calibration_camera_positions = []

        await self.add_calibration_points(pending_calibration_laser_coords)

        if self.logger:
            self.logger.info(
                f"{len(self.calibration_laser_coords)} out of {len(pending_calibration_laser_coords)} point correspondences found."
            )

        if len(self.calibration_laser_coords) < 3:
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

    async def add_calibration_points(self, laser_coords, update_transform=False):
        # TODO: set exposure on camera node automatically when detecting laser
        await self.camera_client.set_exposure(0.001)
        await self.laser_client.set_color((0.0, 0.0, 0.0))
        await self.laser_client.start_laser()
        for laser_coord in laser_coords:
            await self.laser_client.set_point(laser_coord)
            await self.laser_client.set_color(self.laser_color)
            # Wait for galvo to settle and for camera frame capture
            # TODO: increase frame capture rate and reduce this time
            await asyncio.sleep(1)
            await self.add_point_correspondence(laser_coord)
            await self.laser_client.set_color((0.0, 0.0, 0.0))

        await self.laser_client.stop_laser()
        await self.camera_client.set_exposure(-1.0)

        if update_transform:
            self.update_transform_bundle_adjustment()

    def camera_point_to_laser_coord(self, camera_point):
        homogeneous_camera_point = np.hstack((camera_point, 1))
        transformed_point = homogeneous_camera_point @ self.camera_to_laser_transform
        transformed_point = transformed_point / transformed_point[2]
        return transformed_point[:2]

    async def add_point_correspondence(self, laser_coord, num_attempts=3):
        attempts = 0
        while attempts < num_attempts:
            if self.logger:
                self.logger.info(f"attempt = {attempts}")
            attempts += 1
            pos_data = await self.camera_client.get_laser_pos()
            if pos_data["pos_list"]:
                # TODO: handle case where multiple lasers detected
                position = pos_data["pos_list"][0]
                pixel = pos_data["point_list"][0]
                self.calibration_laser_coords.append(laser_coord)
                self.calibration_camera_positions.append(position)
                if self.logger:
                    self.logger.info(
                        f"Found point correspondence: laser_coord = {laser_coord}, pixel = {pixel}, position = {position}. {len(self.calibration_laser_coords)} total correspondences."
                    )
                return
            await asyncio.sleep(0.2)
        if self.logger:
            self.logger.info(
                f"Failed to find point. {len(self.calibration_laser_coords)} total correspondences."
            )

    def update_transform_bundle_adjustment(self):
        def cost_function(parameters, camera_positions, laser_coords):
            homogeneous_camera_positions = np.hstack(
                (camera_positions, np.ones((camera_positions.shape[0], 1)))
            )
            transform = parameters.reshape((4, 3))
            transformed_points = np.dot(homogeneous_camera_positions, transform)
            homogeneous_transformed_points = (
                transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]
            )
            errors = np.sum(
                (homogeneous_transformed_points[:, :2] - laser_coords) ** 2, axis=1
            )
            return np.mean(errors)

        laser_coords = np.array(self.calibration_laser_coords)
        camera_positions = np.array(self.calibration_camera_positions)
        result = minimize(
            cost_function,
            self.camera_to_laser_transform,
            args=(camera_positions, laser_coords),
            method="L-BFGS-B",
        )
        self.camera_to_laser_transform = result.x.reshape((4, 3))

        if self.logger:
            self.logger.info("Updated camera to laser transform")
            self.logger.info(f"Laser coords:\n{laser_coords}")
            homogeneous_camera_positions = np.hstack(
                (camera_positions, np.ones((camera_positions.shape[0], 1)))
            )
            transformed_points = np.dot(
                homogeneous_camera_positions, self.camera_to_laser_transform
            )
            homogeneous_transformed_points = (
                transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]
            )
            self.logger.info(
                f"Laser coords calculated using transform:\n{homogeneous_transformed_points[:, :2]}"
            )
            error = np.mean(
                np.sqrt(
                    np.sum(
                        (homogeneous_transformed_points[:, :2] - laser_coords) ** 2,
                        axis=1,
                    )
                )
            )
            self.logger.info(f"Mean reprojection error: {error}")

    def _update_transform_least_squares(self):
        laser_coords = np.array(self.calibration_laser_coords)
        camera_positions = np.array(self.calibration_camera_positions)
        homogeneous_laser_coords = np.hstack(
            (laser_coords, np.ones((laser_coords.shape[0], 1)))
        )
        homogeneous_camera_positions = np.hstack(
            (camera_positions, np.ones((camera_positions.shape[0], 1)))
        )
        self.camera_to_laser_transform = np.linalg.lstsq(
            homogeneous_camera_positions,
            homogeneous_laser_coords,
            rcond=None,
        )[0]

        if self.logger:
            self.logger.info("Updated camera to laser transform")
            self.logger.info(f"Laser coords:\n{laser_coords}")
            transformed_points = np.dot(
                homogeneous_camera_positions, self.camera_to_laser_transform
            )
            homogeneous_transformed_points = (
                transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]
            )
            self.logger.info(
                f"Laser coords calculated using transform:\n{homogeneous_transformed_points[:, :2]}"
            )
            error = np.mean(
                np.sqrt(
                    np.sum(
                        (homogeneous_transformed_points[:, :2] - laser_coords) ** 2,
                        axis=1,
                    )
                )
            )
            self.logger.info(f"Mean reprojection error: {error}")
