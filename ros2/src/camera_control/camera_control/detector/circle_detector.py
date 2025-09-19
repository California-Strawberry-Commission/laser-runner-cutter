import asyncio
import functools
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np


class CircleDetector:
    def __init__(self, logger: Optional[logging.Logger] = None):
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)

    async def detect(self, color_frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detects centers of circles in a color image.

        Args:
            color_frame (np.ndarray): Color image.
        Returns:
            List[Tuple[int, int]]: List of detected center points.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                self._detect_sync,
                color_frame,
            ),
        )

    def _detect_sync(self, color_frame: np.ndarray) -> List[Tuple[int, int]]:
        # Convert to grayscale
        image = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY)

        # Apply a Gaussian blur to smooth the image and reduce noise
        blurred = cv2.GaussianBlur(image, (9, 9), 2)

        # Use HoughCircles to detect circles in the image
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,  # inverse ratio of resolution (1 means same resolution)
            minDist=50,  # minimum distance between detected centers
            param1=100,  # higher threshold for Canny edge detector
            param2=30,  # accumulator threshold for circle detection
            minRadius=8,  # minimum circle radius
            maxRadius=50,  # maximum circle radius
        )

        circle_centers = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circle_centers = [(x, y) for (x, y, r) in circles]

        return circle_centers
