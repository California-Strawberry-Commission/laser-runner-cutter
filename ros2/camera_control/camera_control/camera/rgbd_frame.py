from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np


class RGBDFrame(ABC):
    color_frame: np.ndarray
    depth_frame: np.ndarray
    timestamp_millis: float
    color_depth_aligned: bool

    @abstractmethod
    def get_position(self, color_pixel: Sequence[int]) -> Optional[List[int]]:
        """
        Given an x-y coordinate in the color frame, return the x-y-z position with respect to the camera.

        Args:
            color_pixel (Sequence[int]): [x, y] coordinate in the color frame.

        Returns:
            Optional[List[int]]: [x, y, z] position with respect to the camera, or None if the position could not be determined
        """
        pass
