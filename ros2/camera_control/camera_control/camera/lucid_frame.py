from typing import Optional, Tuple

import cv2
import numpy as np

from .calibration import distort_pixel_coords, invert_extrinsic_matrix
from .rgbd_frame import RgbdFrame

# General min and max possible depths
DEPTH_MIN_MM = 500
DEPTH_MAX_MM = 10000


class LucidFrame(RgbdFrame):
    color_frame: np.ndarray
    depth_frame: np.ndarray
    timestamp_millis: float

    def __init__(
        self,
        color_frame: np.ndarray,
        depth_frame_xyz: np.ndarray,
        timestamp_millis: float,
        color_camera_intrinsic_matrix: np.ndarray,
        color_camera_distortion_coeffs: np.ndarray,
        depth_camera_intrinsic_matrix: np.ndarray,
        depth_camera_distortion_coeffs: np.ndarray,
        xyz_to_color_camera_extrinsic_matrix: np.ndarray,
        xyz_to_depth_camera_extrinsic_matrix: np.ndarray,
    ):
        """
        Args:
            color_frame (np.ndarray): The color frame in RGB8 format.
            depth_frame_xyz (np.ndarray): The depth frame in Coord3D_ABC16 format.
            timestamp_millis (float): The timestamp of the frame, in milliseconds since the device was started.
            color_camera_intrinsic_matrix (np.ndarray): Intrinsic matrix of the color camera.
            color_camera_distortion_coeffs (np.ndarray): Distortion coefficients of the color camera.
            depth_camera_intrinsic_matrix (np.ndarray): Intrinsic matrix of the depth camera.
            depth_camera_distortion_coeffs (np.ndarray): Distortion coefficients of the depth camera.
            xyz_to_color_camera_extrinsic_matrix (np.ndarray): Extrinsic matrix from depth camera's XYZ positions to the color camera.
            xyz_to_color_camera_extrinsic_matrix (np.ndarray): Extrinsic matrix from depth camera's XYZ positions to the depth camera.
        """
        self.color_frame = color_frame
        self._depth_frame_xyz = depth_frame_xyz
        self.depth_frame = np.sqrt(np.sum(np.square(depth_frame_xyz), axis=-1)).astype(
            np.uint16
        )  # Represent the depth frame as the L2 norm, and convert to mono16

        self.timestamp_millis = timestamp_millis
        self._color_camera_intrinsic_matrix = color_camera_intrinsic_matrix
        self._color_camera_distortion_coeffs = color_camera_distortion_coeffs
        self._depth_camera_intrinsic_matrix = depth_camera_intrinsic_matrix
        self._depth_camera_distortion_coeffs = depth_camera_distortion_coeffs
        self._xyz_to_color_camera_extrinsic_matrix = (
            xyz_to_color_camera_extrinsic_matrix
        )
        self._xyz_to_depth_camera_extrinsic_matrix = (
            xyz_to_depth_camera_extrinsic_matrix
        )
        self._color_to_depth_extrinsic_matrix = (
            xyz_to_depth_camera_extrinsic_matrix
            @ invert_extrinsic_matrix(xyz_to_color_camera_extrinsic_matrix)
        )

    def get_corresponding_depth_pixel_deprecated(
        self, color_pixel: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Given an (x, y) coordinate in the color frame, return the corresponding (x, y) coordinate in the depth frame.

        Args:
            color_pixel (Sequence[int]): (x, y) coordinate in the color frame.

        Returns:
            Tuple[int, int]: (x, y) coordinate in the depth frame.
        """

        # Undistort the pixel coordinate in color camera
        distorted_color_pixel = np.array([[color_pixel]], dtype=np.float32)
        undistorted_color_pixel = cv2.undistortPoints(
            distorted_color_pixel,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
            P=self._color_camera_intrinsic_matrix,
        ).reshape(-1)

        # Convert to normalized camera coordinates
        normalized_color_pixel = np.linalg.inv(
            self._color_camera_intrinsic_matrix
        ) @ np.append(undistorted_color_pixel, 1)
        print(f"normalized_color_pixel = {normalized_color_pixel}")

        # Transform the normalized coordinates to depth camera using the extrinsic matrix
        normalized_color_pixel *= DEPTH_MIN_MM
        transformed_homogeneous_depth_pixel = (
            self._color_to_depth_extrinsic_matrix @ np.append(normalized_color_pixel, 1)
        )
        print(
            f"transformed_homogeneous_depth_pixel = {transformed_homogeneous_depth_pixel}"
        )
        normalized_depth_pixel = (
            transformed_homogeneous_depth_pixel[:3]
            / transformed_homogeneous_depth_pixel[3]
        )
        print(f"normalized_depth_pixel = {normalized_depth_pixel}")

        # Convert to pixel coordinates in depth camera
        undistorted_depth_pixel = (
            self._depth_camera_intrinsic_matrix @ normalized_depth_pixel[:3]
        )
        undistorted_depth_pixel = (
            undistorted_depth_pixel[:2] / undistorted_depth_pixel[2]
        )
        print(f"undistorted_depth_pixel = {undistorted_depth_pixel}")

        # Apply distortion to the pixel coordinates in depth camera
        distorted_depth_pixel = distort_pixel_coords(
            undistorted_depth_pixel,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
        )
        print(f"distorted_depth_pixel = {distorted_depth_pixel}")

        return (
            round(distorted_depth_pixel[0]),
            round(distorted_depth_pixel[1]),
        )

    def get_corresponding_depth_pixel(
        self, color_pixel: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Given an (x, y) coordinate in the color frame, return the corresponding (x, y) coordinate in the depth frame.

        The general approach is as follows:
            1. Deproject the color image pixel coordinate to two positions in the color camera-space: one that corresponds to the
               position at the minimum depth, and one at the maximum depth.
            2. Transform the two positions from color camera-space to depth camera-space.
            3. Project the two positions to their respective depth image pixel coordinates.
            4. The target lies somewhere along the line formed by the two pixel coordinates found in the previous step. We
               iteratively move pixel by pixel along this line. For each depth image pixel, we grab the xyz data at the pixel,
               project it onto the color image plane, and see how far it is from the original color pixel coordinate. We find
               and return the closest match.

        Note that in order to achieve the above, we require two extrinsic matrices - one for projecting the xyz positions to
        the color camera image plane, and one for projecting the xyz positions to the depth camera image plane.

        Args:
            color_pixel (Sequence[int]): (x, y) coordinate in the color frame.

        Returns:
            Tuple[int, int]: (x, y) coordinate in the depth frame.
        """

        def deproject_pixel(pixel, depth, camera_matrix, distortion_coeffs):
            # Normalized, undistorted pixel coord
            pixel = np.array([[[pixel[0], pixel[1]]]], dtype=np.float32)
            pixel_undistorted = cv2.undistortPoints(
                pixel,
                camera_matrix,
                distortion_coeffs,
            )
            return np.array(
                [
                    pixel_undistorted[0][0][0] * depth,
                    pixel_undistorted[0][0][1] * depth,
                    depth,
                ]
            )

        def transform_position(position, extrinsic_matrix):
            position = np.array(position)
            homogeneous_position = np.append(position, 1)
            return np.dot(extrinsic_matrix, homogeneous_position)[:3]

        def project_position(
            position, camera_matrix, distortion_coeffs, extrinsic_matrix=None
        ):
            R = extrinsic_matrix[:3, :3] if extrinsic_matrix is not None else np.eye(3)
            t = extrinsic_matrix[:3, 3] if extrinsic_matrix is not None else np.zeros(3)
            pixels, _ = cv2.projectPoints(
                np.array([[position]]),
                R,
                t,
                camera_matrix,
                distortion_coeffs,
            )
            pixel = pixels[0].flatten()
            return (round(pixel[0]), round(pixel[1]))

        def adjust_pixel_to_bounds(pixel, width, height):
            x = max(0, min(round(pixel[0]), width - 1))
            y = max(0, min(round(pixel[1]), height - 1))
            return (x, y)

        def next_pixel_in_line(curr, start, end):
            # Move one pixel from curr to end
            curr = np.array(curr)
            end = np.array(end)
            direction = end - curr
            direction = direction / np.linalg.norm(direction)
            next = curr + direction
            return (round(next[0]), round(next[1]))

        def is_pixel_in_line(curr, start, end):
            min_x = min(start[0], end[0])
            max_x = max(start[0], end[0])
            min_y = min(start[1], end[1])
            max_y = max(start[1], end[1])

            return min_x <= curr[0] <= max_x and min_y <= curr[1] <= max_y

        # Min-depth and max-depth positions in color camera-space
        min_depth_position_color_space = deproject_pixel(
            color_pixel,
            DEPTH_MIN_MM,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
        )
        max_depth_position_color_space = deproject_pixel(
            color_pixel,
            DEPTH_MAX_MM,
            self._color_camera_intrinsic_matrix,
            self._color_camera_distortion_coeffs,
        )

        # Min-depth and max-depth positions in depth camera-space
        min_depth_position_depth_space = transform_position(
            min_depth_position_color_space, self._color_to_depth_extrinsic_matrix
        )
        max_depth_position_depth_space = transform_position(
            max_depth_position_color_space, self._color_to_depth_extrinsic_matrix
        )

        # Project depth camera-space positions to depth pixels
        min_depth_pixel = project_position(
            min_depth_position_depth_space,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
        )
        max_depth_pixel = project_position(
            max_depth_position_depth_space,
            self._depth_camera_intrinsic_matrix,
            self._depth_camera_distortion_coeffs,
        )

        # Make sure pixel coords are in boundary
        depth_frame_height, depth_frame_width, depth_channels = (
            self._depth_frame_xyz.shape
        )
        min_depth_pixel = adjust_pixel_to_bounds(
            min_depth_pixel, depth_frame_width, depth_frame_height
        )
        max_depth_pixel = adjust_pixel_to_bounds(
            max_depth_pixel, depth_frame_width, depth_frame_height
        )

        # Search along the line for the depth pixel for which its corresponding projected color pixel is the closest
        # to the target color pixel
        min_dist = -1
        closest_depth_pixel = min_depth_pixel
        curr_depth_pixel = min_depth_pixel
        while True:
            xyz_mm = self._depth_frame_xyz[curr_depth_pixel[1]][curr_depth_pixel[0]]
            curr_color_pixel = project_position(
                xyz_mm,
                self._color_camera_intrinsic_matrix,
                self._color_camera_distortion_coeffs,
                self._xyz_to_color_camera_extrinsic_matrix,
            )
            distance = np.linalg.norm(
                np.array(curr_color_pixel).astype(float)
                - np.array(color_pixel).astype(float)
            )
            if distance < min_dist or min_dist < 0:
                min_dist = distance
                closest_depth_pixel = curr_depth_pixel

            # Stop if we've processed the max_depth_pixel
            if (
                curr_depth_pixel[0] == max_depth_pixel[0]
                and curr_depth_pixel[1] == max_depth_pixel[1]
            ):
                break

            # Otherwise, find the next pixel along the line we should try
            curr_depth_pixel = next_pixel_in_line(
                curr_depth_pixel, min_depth_pixel, max_depth_pixel
            )
            if not is_pixel_in_line(curr_depth_pixel, min_depth_pixel, max_depth_pixel):
                break

        return closest_depth_pixel

    def get_position(
        self, color_pixel: Tuple[int, int]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Given an (x, y) coordinate in the color frame, return the (x, y, z) position with respect to the camera.

        Args:
            color_pixel (Sequence[int]): (x, y) coordinate in the color frame.

        Returns:
            Optional[Tuple[float, float, float]]: (x, y, z) position with respect to the camera, or None if the position could not be determined.
        """
        depth_pixel = self.get_corresponding_depth_pixel(color_pixel)
        position = self._depth_frame_xyz[depth_pixel[1]][depth_pixel[0]]
        # Negative depth indicates an invalid position
        if position[2] < 0.0:
            return None

        return (float(position[0]), float(position[1]), float(position[2]))
