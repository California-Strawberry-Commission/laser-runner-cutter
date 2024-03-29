import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor

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
        self.calibration_camera_points = []
        self.is_calibrated = False
        self.camera_to_laser_transform = np.zeros((4, 3))

    async def calibrate(self):
        """
        Use image correspondences to compute the transformation matrix from camera to laser.
        Note that calling this resets the point correspondences.

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
        self.calibration_camera_points = []

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
        await self.update_transform_bundle_adjustment()

        self.is_calibrated = True
        return True

    async def add_calibration_points(self, laser_coords, update_transform=False):
        """
        Find and add additional point correspondences by shooting the laser at each laser_coords
        and then optionally recalculate the transform.

        Args:
            laser_coords: [(x: float, y: float)], laser coordinates to find point correspondences with
            update_transform: bool, whether to recalculate the camera-space position to laser coord transform
        """

        # TODO: set exposure on camera node automatically when detecting laser
        await self.camera_client.set_exposure(0.001)
        await self.laser_client.set_color((0.0, 0.0, 0.0))
        await self.laser_client.start_laser()
        for laser_coord in laser_coords:
            await self.laser_client.set_point(laser_coord)
            await self.laser_client.set_color(self.laser_color)
            # Wait for galvo to settle and for camera frame capture
            await asyncio.sleep(0.1)
            camera_point = await self._find_point_correspondence(laser_coord)
            if camera_point is not None:
                await self.add_point_correspondence(laser_coord, camera_point)
            # We use set_color instead of stop_laser as it is faster to temporarily turn off the laser
            await self.laser_client.set_color((0.0, 0.0, 0.0))

        await self.laser_client.stop_laser()
        await self.camera_client.auto_exposure()

        if update_transform:
            await self.update_transform_bundle_adjustment()

    def camera_point_to_laser_coord(self, camera_point):
        """
        Transform a 3D point in camera-space to a laser coord

        Args:
            camera_point: [(x: float, y: float, z: float)], a point in camera-space
        """
        if not self.is_calibrated:
            return None

        homogeneous_camera_point = np.hstack((camera_point, 1))
        transformed_point = homogeneous_camera_point @ self.camera_to_laser_transform
        transformed_point = transformed_point / transformed_point[2]
        return transformed_point[:2]

    async def _find_point_correspondence(self, laser_coord, num_attempts=3):
        """
        For the given laser coord, find the corresponding 3D point in camera-space.

        Args:
            laser_coord: [(x: float, y: float)], laser coordinate to find point correspondence for
        """
        attempt = 0
        while attempt < num_attempts:
            if self.logger:
                self.logger.info(
                    f"Attempt {attempt} to detect laser and find point correspondence."
                )
            attempt += 1
            pos_data = await self.camera_client.get_laser_pos()
            if pos_data["pos_list"]:
                # TODO: handle case where multiple lasers detected
                position = pos_data["pos_list"][0]
                pixel = pos_data["point_list"][0]
                if self.logger:
                    self.logger.info(
                        f"Found point correspondence: laser_coord = {laser_coord}, pixel = {pixel}, position = {position}."
                    )
                return position
            await asyncio.sleep(0.2)
        if self.logger:
            self.logger.info(
                f"Failed to find point. {len(self.calibration_laser_coords)} total correspondences."
            )
        return None

    async def add_point_correspondence(
        self, laser_coord, camera_point, update_transform=False
    ):
        self.calibration_laser_coords.append(laser_coord)
        self.calibration_camera_points.append(camera_point)
        if self.logger:
            self.logger.info(
                f"Added point correspondence. {len(self.calibration_laser_coords)} total correspondences."
            )

        if update_transform:
            await self.update_transform_bundle_adjustment()

    async def update_transform_bundle_adjustment(self):
        def cost_function(parameters, camera_points, laser_coords):
            homogeneous_camera_points = np.hstack(
                (camera_points, np.ones((camera_points.shape[0], 1)))
            )
            transform = parameters.reshape((4, 3))
            transformed_points = np.dot(homogeneous_camera_points, transform)
            homogeneous_transformed_points = (
                transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]
            )
            errors = np.sum(
                (homogeneous_transformed_points[:, :2] - laser_coords) ** 2, axis=1
            )
            return np.mean(errors)

        laser_coords = np.array(self.calibration_laser_coords)
        camera_points = np.array(self.calibration_camera_points)

        with ThreadPoolExecutor() as executor:
            result = await asyncio.get_event_loop().run_in_executor(
                executor,
                functools.partial(
                    minimize,
                    cost_function,
                    self.camera_to_laser_transform,
                    args=(camera_points, laser_coords),
                    method="L-BFGS-B",
                ),
            )

        self.camera_to_laser_transform = result.x.reshape((4, 3))

        if self.logger:
            self.logger.info(
                "Updated camera to laser transform using bundle adjustment"
            )
            self.logger.info(f"Laser coords:\n{laser_coords}")
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
        camera_points = np.array(self.calibration_camera_points)
        homogeneous_laser_coords = np.hstack(
            (laser_coords, np.ones((laser_coords.shape[0], 1)))
        )
        homogeneous_camera_points = np.hstack(
            (camera_points, np.ones((camera_points.shape[0], 1)))
        )
        self.camera_to_laser_transform = np.linalg.lstsq(
            homogeneous_camera_points,
            homogeneous_laser_coords,
            rcond=None,
        )[0]

        if self.logger:
            self.logger.info("Updated camera to laser transform using least squares")
            self.logger.info(f"Laser coords:\n{laser_coords}")
            transformed_points = np.dot(
                homogeneous_camera_points, self.camera_to_laser_transform
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
