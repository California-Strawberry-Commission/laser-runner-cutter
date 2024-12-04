from __future__ import annotations
import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np

DEPTH_HFOV = math.radians(87)  # Realsense - 87 degrees
IN_TO_MM = 25.4
FURROW_MIN_WIDTH = 2 * IN_TO_MM
FURROW_MAX_WIDTH = 20 * IN_TO_MM


class FurrowStripTracker:

    def __init__(
        self, strip_height: int, strip_idx: int, overall_dim: Tuple[int, int]
    ) -> None:
        self._height, self._width = overall_dim
        self._rad_per_px = DEPTH_HFOV / self._width

        self._idx = strip_idx
        self._strip_height = strip_height
        self._kernel = (strip_height, strip_height)
        self._search_min, self._search_max = 0, self._width
        self._bound_deadband = strip_height // 2 + 1

        self._is_valid = False
        self._convolution = []
        self._y_min = 0
        self._y_max = 0
        self._y_center = 0
        self._x_center = 0
        self._left_bound = 0
        self._right_bound = 0

        self._reference_distance = 0
        self._furrow_width = 0.0
        self._delta_x_from_previous = 0
        self._time_convolve_strip = 0.0
        self._time_find_bounds = 0.0

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    @property
    def y_center(self) -> int:
        return self._y_center

    @property
    def x_center(self) -> int:
        return self._x_center

    @property
    def left_bound(self) -> int:
        return self._left_bound

    @property
    def right_bound(self) -> int:
        return self._right_bound

    def process(self, depth_img, previous_tracker: Optional[FurrowStripTracker]):
        """Index is from bottom of image"""
        t = time.perf_counter()
        self._convolution, self._y_min, self._y_max, self._y_center = (
            self.convolve_strip(depth_img)
        )
        self._time_convolve_strip = time.perf_counter() - t

        t = time.perf_counter()
        self._left_bound, self._right_bound = self.find_bounds(self._convolution)
        self._x_center = self._left_bound + (self._right_bound - self._left_bound) // 2
        self._time_find_bounds = time.perf_counter() - t

        self.process_results(previous_tracker)
        self._is_valid = self.check_validity()

    def convolve_strip(self, depth_img):
        strip_bottom = self._height - (self._idx * self._strip_height)
        strip_top = strip_bottom - self._strip_height
        roi = depth_img[strip_top:strip_bottom]
        # print(self.height, self.strip_height, strip_top, strip_bottom, roi)

        # print(self.kernel)
        # convolution = convolve(roi, self.kernel, mode="valid")[0]
        # convolution = cv2.filter2D(src=roi, kernel=self.kernel, ddepth=-1, borderType=cv2.BORDER_ISOLATED)[0]
        # convolution = cv2.blur(src=roi, kernel=self.kernel, ddepth=-1, borderType=cv2.BORDER_ISOLATED)[0]
        # convolution = cv2.stackBlur(roi, self.kernel)
        convolution = cv2.blur(
            roi,
            (self._strip_height, self._strip_height),
            borderType=cv2.BORDER_ISOLATED,
        )[0]
        # print(convolution)
        # FFT convolve the ROI with a box kernel to smooth depth signal
        return (
            convolution,
            strip_top,
            strip_bottom,
            (strip_top + (strip_bottom - strip_top) // 2),
        )

    def find_bounds(
        self,
        convolution,
        thresh_factor: float = 0.2,
        search_start: Optional[int] = None,
    ):
        search_start = search_start or len(convolution) // 2

        self._reference_distance = convolution[search_start]
        threshold = int(self._reference_distance * (1 - thresh_factor))

        l_bound = self._search_min + np.searchsorted(
            convolution[self._search_min : search_start], threshold, side="right"
        )

        # Reverse array before searching right bound -
        # searchsorted needs ascending vals
        r_bound = self._search_max - np.searchsorted(
            convolution[self._search_max : search_start : -1], threshold, side="right"
        )

        return l_bound, r_bound

    def process_results(self, previous_tracker: Optional[FurrowStripTracker]):
        # Calculate detected width of the furrow
        rad = (self._right_bound - self._left_bound) * self._rad_per_px
        self._furrow_width = abs(rad * self._reference_distance)

        if previous_tracker is not None:
            self._delta_x_from_previous = self.x_center - previous_tracker.x_center

    def check_validity(self) -> bool:
        # Check if search found furrow wall
        if (
            self._left_bound <= self._search_min + self._bound_deadband
            or self._right_bound >= self._search_max - self._bound_deadband
        ):
            return False

        # Check that detected width is within bounds of an expected typical furrow
        if not FURROW_MIN_WIDTH <= self._furrow_width <= FURROW_MAX_WIDTH:
            return False

        return True
