import time
from typing import List, Optional

import numpy as np
from scipy.stats import linregress

from .furrow_strip_tracker import FurrowStripTracker


class FurrowTracker:
    def __init__(self, num_strips: int = 10):
        self._num_strips = num_strips
        self._last_process_time = 0.0
        self._strips: List[FurrowStripTracker] = []

        self._width: int = 0
        self._height: int = 0

        self._reg_slope: Optional[float] = 0
        self._reg_intercept: Optional[float] = 0

        self._pin_y = 0

        self.guidance_offset_x = -40

    @property
    def last_process_time(self) -> float:
        return self._last_process_time

    @property
    def strips(self) -> List[FurrowStripTracker]:
        return self._strips

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def reg_slope(self) -> Optional[float]:
        return self._reg_slope

    @property
    def reg_intercept(self) -> Optional[float]:
        return self._reg_intercept

    @property
    def pin_y(self) -> int:
        return self._pin_y

    def init(self, depth_img):
        if self._width and self._height:
            return
        depth_img = np.asarray(depth_img)
        depth_shape = depth_img.shape
        self._height, self._width = depth_shape
        self.strip_height = self._height // self._num_strips

        self._pin_y = 2 * self._height // 4

        for i in range(self._num_strips):
            self._strips.append(FurrowStripTracker(self.strip_height, i, depth_shape))

    def process(self, depth_img):
        self.init(depth_img)

        start = time.perf_counter()

        prev = None
        for s in self._strips:
            s.process(depth_img, prev)
            prev = s

        self.regress_strips()

        self._last_process_time = time.perf_counter() - start

    def regress_strips(self):
        """Regresses current centerpoints
        NOTE: Resulting regression is transposed to avoid infinite slope"""
        if not self._width and not self._height:
            return

        self._reg_slope = None
        self._reg_intercept = None

        valid_centerpoints = [
            (s.x_center, s.y_center) for s in self._strips if s.is_valid
        ]

        if len(valid_centerpoints) <= 2:
            return

        x, y = zip(*valid_centerpoints)

        if np.all(np.asarray(x) == x[0]):
            return

        regression = linregress(y, x)

        self._reg_slope = regression.slope
        self._reg_intercept = regression.intercept

    def get_reg_x(self, y: int) -> Optional[int]:
        if self._reg_intercept is not None:
            return int(y * (self._reg_slope or 1) + self._reg_intercept)
        return None

    def get_error(self):
        guidance_x = self._width // 2 + self.guidance_offset_x

        if pin_x := self.get_reg_x(self._pin_y):
            return guidance_x - pin_x

        return None
