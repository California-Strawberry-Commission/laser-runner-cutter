import asyncio
import functools
import logging
import pickle
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, minimize

from camera_control.camera_control_node import CameraControlNode
from common_interfaces.msg import Vector2
from laser_control.laser_control_node import LaserControlNode
from runner_cutter_control.camera_context import CameraContext

CALIBRATION_FILENAME = "calibration.pkl"


class PointCorrespondences:
    camera_to_laser_transform: np.ndarray
    camera_frame_size: Tuple[int, int]
    # The rect (min x, min y, width, height) representing the reach of the laser, in terms of camera pixels.
    laser_bounds: Tuple[int, int, int, int]

    def __init__(self):
        self._laser_coords = []
        self._camera_pixels = []
        self._camera_points = []
        self.camera_to_laser_transform = np.zeros((4, 3))
        self.camera_frame_size = (0, 0)
        self.laser_bounds = (0, 0, 0, 0)

    def __len__(self):
        return len(self._laser_coords)

    def add(
        self,
        laser_coord: Tuple[float, float],
        camera_pixel: Tuple[int, int],
        camera_point: Tuple[float, float, float],
    ):
        self._laser_coords.append(laser_coord)
        self._camera_pixels.append(camera_pixel)
        self._camera_points.append(camera_point)
        self._update_laser_bounds()

    def clear(self):
        self._laser_coords.clear()
        self._camera_pixels.clear()
        self._camera_points.clear()
        self.camera_to_laser_transform = np.zeros((4, 3))
        self.camera_frame_size = (0, 0)
        self.laser_bounds = (0, 0, 0, 0)

    async def update_transform_nonlinear_least_squares(self):
        def residuals(parameters, camera_points, laser_coords):
            homogeneous_camera_points = np.hstack(
                (camera_points, np.ones((camera_points.shape[0], 1)))
            )
            transform = parameters.reshape((4, 3))
            transformed_points = np.dot(homogeneous_camera_points, transform)
            homogeneous_transformed_points = (
                transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]
            )
            return (homogeneous_transformed_points[:, :2] - laser_coords).flatten()

        camera_points = np.array(self._camera_points)
        laser_coords = np.array(self._laser_coords)

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                least_squares,
                residuals,
                self.camera_to_laser_transform.flatten(),
                args=(camera_points, laser_coords),
                method="trf",
            ),
        )

        self.camera_to_laser_transform = result.x.reshape((4, 3))

    def update_transform_linear_least_squares(self):
        camera_points = np.array(self._camera_points)
        laser_coords = np.array(self._laser_coords)
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

    def get_reprojection_error(self) -> float:
        camera_points = np.array(self._camera_points)
        laser_coords = np.array(self._laser_coords)
        homogeneous_camera_points = np.hstack(
            (camera_points, np.ones((camera_points.shape[0], 1)))
        )
        transformed_points = np.dot(
            homogeneous_camera_points, self.camera_to_laser_transform
        )
        homogeneous_transformed_points = (
            transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]
        )
        reprojection_error = np.mean(
            np.sqrt(
                np.sum(
                    (homogeneous_transformed_points[:, :2] - laser_coords) ** 2,
                    axis=1,
                )
            )
        )
        return reprojection_error

    def _update_laser_bounds(self):
        x_values = [x for x, _ in self._camera_pixels]
        y_values = [y for _, y in self._camera_pixels]
        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)
        self.laser_bounds = (min_x, min_y, max_x - min_x, max_y - min_y)


class Calibration:
    is_calibrated: bool

    def __init__(
        self,
        laser_node: LaserControlNode,
        camera_node: CameraControlNode,
        laser_color: Tuple[float, float, float],
        logger: Optional[logging.Logger] = None,
    ):
        self._laser_node = laser_node
        self._camera_node = camera_node
        self._camera_context = CameraContext(camera_node)
        self._laser_color = laser_color
        self._point_correspondences = PointCorrespondences()
        self.is_calibrated = False
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)

    @property
    def camera_frame_size(self) -> Tuple[int, int]:
        return self._point_correspondences.camera_frame_size

    @property
    def laser_bounds(self) -> Tuple[int, int, int, int]:
        return self._point_correspondences.laser_bounds

    @property
    def normalized_laser_bounds(self) -> Tuple[float, float, float, float]:
        return (
            (
                self.laser_bounds[0] / self.camera_frame_size[0]
                if self.camera_frame_size[0] > 0
                else 0.0
            ),
            (
                self.laser_bounds[1] / self.camera_frame_size[1]
                if self.camera_frame_size[1] > 0
                else 0.0
            ),
            (
                self.laser_bounds[2] / self.camera_frame_size[0]
                if self.camera_frame_size[0] > 0
                else 0.0
            ),
            (
                self.laser_bounds[3] / self.camera_frame_size[1]
                if self.camera_frame_size[1] > 0
                else 0.0
            ),
        )

    def reset(self):
        self._point_correspondences.clear()
        self.is_calibrated = False

    def save(self, dir: str) -> Optional[str]:
        if not self.is_calibrated:
            return None

        save_dir = os.path.expanduser(dir)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, CALIBRATION_FILENAME)
        with open(filepath, "wb") as f:
            pickle.dump(self._point_correspondences, f)

        self._logger.info(f"Calibration saved to {filepath}")
        return filepath

    def load(self, dir: str) -> Optional[str]:
        save_dir = os.path.expanduser(dir)
        filepath = os.path.join(save_dir, CALIBRATION_FILENAME)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as file:
                    self._point_correspondences = pickle.load(file)
                    self.is_calibrated = True
                self._logger.info(f"Successfully loaded calibration file {filepath}")
                return filepath
            except Exception as e:
                self._logger.warning(f"Could not load calibration file {filepath}: {e}")
                return None
        else:
            self._logger.info(f"Could not find calibration file {filepath}")
            return None

    async def calibrate(self, grid_size: Tuple[int, int] = (5, 5)) -> bool:
        """
        Use image correspondences to compute the transformation matrix from camera to laser.
        Note that calling this resets the point correspondences.

        1. Shoot the laser at predetermined points
        2. For each laser point, capture an image from the camera
        3. Identify the corresponding point in the camera frame
        4. Compute the transformation matrix from camera to laser

        Args:
            grid_size (Tuple[int, int]): Number of points in the x and y directions to use as calibration points.
        Returns:
            bool: Whether calibration was successful or not.
        """
        self.reset()

        # TODO: On RealSense, depth values are noisy when the laser is on. Figure out how to reduce noise,
        # or use a depth frame obtained when the laser is off

        # Get camera color frame size
        frame = await self._camera_node.get_frame()
        self._point_correspondences.camera_frame_size = (
            frame.color_frame.width,
            frame.color_frame.height,
        )

        # Get calibration points
        x_step = 1.0 / (grid_size[0] - 1)
        y_step = 1.0 / (grid_size[1] - 1)
        pending_calibration_laser_coords: List[Tuple[float, float]] = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x = i * x_step
                y = j * y_step
                pending_calibration_laser_coords.append((x, y))

        # Get image correspondences
        self._logger.info("Get image correspondences")
        await self.add_calibration_points(pending_calibration_laser_coords)

        self._logger.info(
            f"{len(self._point_correspondences)} out of {len(pending_calibration_laser_coords)} point correspondences found."
        )

        if len(self._point_correspondences) < 3:
            self._logger.warning(
                "Calibration failed: insufficient point correspondences found."
            )
            return False

        # Use linear least squares for an initial estimate, then refine using nonlinear least squares
        self._point_correspondences.update_transform_linear_least_squares()
        await self._point_correspondences.update_transform_nonlinear_least_squares()

        self.is_calibrated = True
        return True

    async def add_calibration_points(
        self, laser_coords: List[Tuple[float, float]], update_transform: bool = False
    ):
        """
        Find and add additional point correspondences by shooting the laser at each laser_coords
        and then optionally recalculate the transform.

        Args:
            laser_coords (List[Tuple[float, float]]): Laser coordinates to find point correspondences with
            update_transform (bool): Whether to recalculate the camera-space position to laser coord transform
        """

        # TODO: set exposure/gain on camera node automatically when detecting laser
        async with self._camera_context.laser_detection_settings():
            await self._laser_node.clear_points()
            await self._laser_node.set_color(r=0.0, g=0.0, b=0.0, i=0.0)
            try:
                await self._laser_node.play()
                for laser_coord in laser_coords:
                    await self._laser_node.set_points(
                        points=[Vector2(x=laser_coord[0], y=laser_coord[1])]
                    )
                    await self._laser_node.set_color(
                        r=self._laser_color[0],
                        g=self._laser_color[1],
                        b=self._laser_color[2],
                        i=0.0,
                    )
                    # Wait for galvo to settle and for camera frame capture
                    # TODO: optimize the frame callback time and reduce this
                    await asyncio.sleep(0.5)
                    camera_pixel, camera_point = await self._find_point_correspondence(
                        laser_coord, attempt_interval_s=0.25
                    )
                    if camera_pixel is not None and camera_point is not None:
                        await self.add_point_correspondence(
                            laser_coord, camera_pixel, camera_point
                        )
                    # We use set_color() instead of stop() as it is faster to temporarily turn off the laser
                    await self._laser_node.set_color(r=0.0, g=0.0, b=0.0, i=0.0)
            finally:
                await self._laser_node.stop()

        if update_transform:
            await self._point_correspondences.update_transform_nonlinear_least_squares()

    def camera_point_to_laser_coord(
        self, position: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """
        Transform a 3D position in camera-space to a laser coord.

        Args:
            position (Tuple[float, float, float]): A 3D position (x, y, z) in camera-space.
        Returns:
            Tuple[float, float]: (x, y) laser coordinates.
        """
        homogeneous_camera_point = np.hstack((position, 1))
        transformed_point = (
            homogeneous_camera_point
            @ self._point_correspondences.camera_to_laser_transform
        )
        try:
            transformed_point = transformed_point / transformed_point[2]
            return (transformed_point[0], transformed_point[1])
        except:
            return (-1.0, -1.0)

    async def _find_point_correspondence(
        self,
        laser_coord: Tuple[float, float],
        num_attempts: int = 3,
        attempt_interval_s: float = 0.2,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[float, float, float]]]:
        """
        For the given laser coord, find the corresponding 3D point in camera-space.

        Args:
            laser_coord (Tuple[float, float]): Laser coordinate (x, y) to find point correspondence for.
            num_attempts (int): Number of tries to detect the laser and find the point correspondence.
        Returns:
            Tuple[Optional[Tuple[int, int]], Optional[Tuple[float, float, float]]]: (pixel coord in camera frame, 3D position in camera-space), or (None, None) if the laser could not be detected.
        """
        attempt = 0
        while attempt < num_attempts:
            self._logger.info(
                f"Attempt {attempt} to detect laser and find point correspondence."
            )
            attempt += 1
            result = await self._camera_node.get_laser_detection()
            detection_result = result.result
            instances = detection_result.instances
            if instances:
                # TODO: handle case where multiple lasers detected
                instance = instances[0]
                self._logger.info(
                    f"Found point correspondence: laser_coord = {laser_coord}, pixel = {instance.point}, position = {instance.position}."
                )
                return (
                    (round(instance.point.x), round(instance.point.y)),
                    (instance.position.x, instance.position.y, instance.position.z),
                )
            await asyncio.sleep(attempt_interval_s)
        self._logger.info(
            f"Failed to find point. {len(self._point_correspondences)} total correspondences."
        )
        return (None, None)

    async def add_point_correspondence(
        self,
        laser_coord: Tuple[float, float],
        camera_pixel: Tuple[int, int],
        camera_point: Tuple[float, float, float],
        update_transform: bool = False,
    ):
        """
        Add the point correspondence between laser coord and camera-space position and optionally
        update the transform.

        Args:
            laser_coord (Tuple[float, float]): Laser coordinate (x, y) of the point correspondence.
            camera_point (Tuple[float, float, float]): Camera-space position (x, y, z) of the point correspondence.
            update_transform (bool): Whether to update the transform matrix or not.
        """
        self._point_correspondences.add(laser_coord, camera_pixel, camera_point)
        self._logger.info(
            f"Added point correspondence. {len(self._point_correspondences)} total correspondences."
        )

        # This is behind a flag as updating the transform is computationally non-trivial
        if update_transform:
            await self._point_correspondences.update_transform_nonlinear_least_squares()
