import time
from typing import List
import numpy as np
import time
from scipy.stats import linregress
from .furrow_strip_tracker import FurrowStripTracker

MARKER_SIZE = 10
DISPLAY_LINES = True
BG_WIDTH = 140


class FurrowTracker:
    last_process_time = 0
    strips: List[FurrowStripTracker] = []

    width: int = 0
    height: int = 0

    reg_slope = 0
    reg_intercept = 0

    pin_y = 0
    guidance_offset_x = -40
    is_controlling_angular = 0

    def __init__(self, num_strips=10):
        self.num_strips = num_strips

    def init(self, depth_img):
        if self.width and self.height:
            return
        depth_img = np.asarray(depth_img)
        depth_shape = depth_img.shape
        self.height, self.width = depth_shape
        self.strip_height = self.height // self.num_strips

        self.pin_y = 2 * self.height // 4

        for i in range(self.num_strips):
            self.strips.append(FurrowStripTracker(
                self.strip_height, i, depth_shape))

    def process(self, depth_img):
        self.init(depth_img)

        start = time.perf_counter()

        prev = None
        for s in self.strips:
            s.process(depth_img, prev)
            prev = s

        self.regress_strips()

        self.last_process_time = time.perf_counter() - start

    def regress_strips(self):
        """Regresses current centerpoints 
        NOTE: Resulting regression is transposed to avoid infinite slope"""
        if not self.width and not self.height:
            return

        self.reg_slope = None
        self.reg_intercept = None

        valid_centerpoints = [
            (s.x_center, s.y_center) for s in self.strips if s.is_valid
        ]

        if len(valid_centerpoints) <= 2:
            return

        x, y = zip(*valid_centerpoints)

        if np.all(np.asarray(x) == x[0]):
            return

        regression = linregress(y, x)

        self.reg_slope = regression.slope
        self.reg_intercept = regression.intercept

    def get_reg_x(self, y):
        if self.reg_intercept is not None:
            return int(y * (self.reg_slope or 1) + self.reg_intercept)
        return None

    def get_error(self):
        guidance_x = self.width // 2 + self.guidance_offset_x

        if pin_x := self.get_reg_x(self.pin_y):
            return guidance_x - pin_x

        return None

    def set_angular_control(self, state):
        self.is_controlling_angular = state
