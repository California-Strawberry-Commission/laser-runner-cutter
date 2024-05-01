from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class RgbdFrame(ABC):
    color_frame: np.ndarray
    depth_frame: np.ndarray
    timestamp_millis: float
    color_depth_aligned: bool

    @abstractmethod
    def get_position(
        self, color_pixel: Tuple[int, int]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Given an x-y coordinate in the color frame, return the x-y-z position with respect to the camera.

        Args:
            color_pixel (Tuple[int, int]): (x, y) coordinate in the color frame.

        Returns:
            Optional[Tuple[float, float, float]]: (x, y, z) position with respect to the camera, or None if the position could not be determined.
        """
        pass
