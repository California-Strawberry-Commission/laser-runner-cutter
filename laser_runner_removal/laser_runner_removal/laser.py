"""File: laser.py

Class for controlling an ILDA laser through a HeliosDac. This class operates by adding 
points to a point list and then sending the list with a call to sendFrame. 
When adding a points to the laser frame color, pad, and intensity need to be added. 
-Color: Effects both color and power, (255, 0, 0) will be higher power then (10, 0, 0), but the same color. 
-Pad: Adds points before and after the adding points, this is needed so the laser will create clean
    points and lines, instead of blurring between added points. 
-Intensity: Currently does not have a noticeable effect on any lasers currently tested, 
    but is included in HelsiosDac functions. 

Utilizes the HeliosDac code from https://github.com/Grix/helios_dac
"""
import numpy as np

import ctypes
import os
import logging

from ament_index_python.packages import get_package_share_directory


# Define point structure for HeliosDac
class HeliosPoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint16),
        ("y", ctypes.c_uint16),
        ("r", ctypes.c_uint8),
        ("g", ctypes.c_uint8),
        ("b", ctypes.c_uint8),
        ("i", ctypes.c_uint8),
    ]


class IldaLaser:
    def __init__(self, frame_shape=(4096, 4096), logger=None):
        self.frame_shape = frame_shape

        # List of points that will get sent to the laser as a frame when sendFrame is called
        self.point_list = []

        # Transform from world space to laser image space
        self.transform_to_laser = None

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger()

    @property
    def frame_corners(self):
        """Return the corners of the frame based on a 0,0 origin."""
        x = self.frame_shape[0]
        y = self.frame_shape[1]
        return np.array([[0, 0], [x, 0], [x, y], [0, y]])

    def scale_frame_corners(self, scale_factor):
        """Scale the frame around the center based on a scale factor.

        Scale factor should be between 0 and 1.
        """
        # Calculate the center of the rectangle
        center_x = self.frame_shape[0] / 2
        center_y = self.frame_shape[1] / 2

        # Calculate the new coordinates of the corners
        new_x = self.frame_shape[0] * scale_factor
        new_y = self.frame_shape[1] * scale_factor

        new_corners = [
            [center_x - new_x / 2, center_y - new_y / 2],  # Top-left corner
            [center_x + new_x / 2, center_y - new_y / 2],  # Top-right corner
            [center_x + new_x / 2, center_y + new_y / 2],  # Bottom-right corner
            [center_x - new_x / 2, center_y + new_y / 2],  # Bottom-left corner
        ]

        return new_corners

    def initialize(self):
        """Create connection to a ILDA camera"""
        # Load and initialize library
        include_dir = os.path.join(
            get_package_share_directory("laser_control"), "include"
        )
        self.HeliosLib = ctypes.cdll.LoadLibrary(
            os.path.join(include_dir, "libHeliosDacAPI.so")
        )
        numDevices = self.HeliosLib.OpenDevices()
        self.logger.info(f"Found {numDevices} Helios DACs")

    def close(self):
        self.HeliosLib.CloseDevices()

    def add_pos(self, pos, color=(10, 0, 0), pad=50, intensity=1):
        """Add a global position to the laser frame"""
        point = pos @ self.transform_to_laser
        self.logger.debug(f"sending point:{point} for pos: {pos}")
        self.add_point(point, color=color, pad=pad, intensity=intensity)
        return point

    def add_point(self, point, color=(10, 0, 0), pad=50, intensity=1):
        """Add a point to the laser frame"""
        if point[0] > 4095:
            self.logger.warning("WARNING laser X > 4095")
            point[0] = 4095
        if point[1] > 4095:
            self.logger.warning("WARNING laser Y > 4095")
            point[1] = 4095
        if point[0] < 0:
            self.logger.warning("WARNING laser X < 0")
            point[0] = 0
        if point[1] < 0:
            self.logger.warning("WARNING laser Y > 0")
            point[1] = 0
        if pad:
            for x in range(pad):
                self.point_list.append(
                    HeliosPoint(int(point[0]), int(point[1]), 0, 0, 0, 0)
                )
        self.point_list.append(
            HeliosPoint(
                int(point[0]), int(point[1]), color[0], color[1], color[2], intensity
            )
        )
        if pad:
            for x in range(pad):
                self.point_list.append(
                    HeliosPoint(int(point[0]), int(point[1]), 0, 0, 0, 0)
                )

    def add_square(self, center, radius, color=(10, 0, 0), pad=50):
        """Add a square to the laser frame. The center must be in normalized coordinates"""
        for x in range(-1 * radius, radius + 1):
            for y in range(-1 * radius, radius + 1):
                self.point_list.append(
                    HeliosPoint(
                        x + int(center[0]),
                        y + int(center[1]),
                        color[0],
                        color[1],
                        color[2],
                        0,
                    )
                )
        if pad:
            for x in range(100):
                self.point_list.append(
                    HeliosPoint(int(center[0]), int(center[1]), 0, 0, 0, 0)
                )

    def add_edges(self):
        """Add spaced out points to create edges in the laser frame."""
        for x in range(32):
            self.point_list.append(HeliosPoint(x * 128, 0, 255, 0, 0, 0))
        for y in range(32):
            self.point_list.append(HeliosPoint(4095, y * 128, 255, 0, 0, 0))
        for x in range(32):
            self.point_list.append(HeliosPoint(4095 - x * 128, 4095, 255, 0, 0, 0))
        for y in range(32):
            self.point_list.append(HeliosPoint(0, 4095 - y * 128, 255, 0, 0, 0))

    def sendFrame(self, verbose=False):
        """Sends the current list of points to the Laser."""
        num_points = len(self.point_list)
        frame_cls = HeliosPoint * num_points
        frame = frame_cls()
        for idx, point in enumerate(self.point_list):
            frame[idx] = point

        status = self.HeliosLib.GetStatus(0)
        statusAttempts = 0
        while statusAttempts < 512 and status != 1:
            statusAttempts += 1
            if verbose:
                self.logger.debug(
                    f"Status check {statusAttempts} failed, status :{status}"
                )
            status = self.HeliosLib.GetStatus(0)

        ret = self.HeliosLib.WriteFrame(
            0, num_points * 45, 0, ctypes.pointer(frame), num_points
        )  # Send the frame
        self.point_list = []

    def sendEmpty(self, x=None, y=None):
        """Reset the point list and send a point with no color to the laser."""
        if x is None:
            x = self.frame_shape[0] / 2
        if y is None:
            y = self.frame_shape[1] / 2
        self.point_list = []
        for x in range(10):
            self.point_list.append(HeliosPoint(int(x), int(y), 0, 0, 0, 0))
        self.sendFrame()
