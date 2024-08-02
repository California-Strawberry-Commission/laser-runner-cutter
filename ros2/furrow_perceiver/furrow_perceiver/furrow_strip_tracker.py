import time
from typing import List
import numpy as np
import cv2
import time
from scipy.stats import linregress

MARKER_SIZE = 10
DISPLAY_LINES = True
BG_WIDTH = 140

class FurrowStripTracker:
    DEPTH_HFOV = np.deg2rad(87)  # Realsense - 87 degrees
    FURROW_MIN_WIDTH = 2 * 25.4  # 10 inches
    FURROW_MAX_WIDTH = 20 * 25.4  # 20 inches

    convolution = []
    ymin = 0
    ymax = 0
    y_center = 0
    x_center = 0
    
    left_bound = 0
    right_bound = 0
    bound_deadband = 3

    reference_distance = 0
    furrow_width = 0
    delta_x_from_previous = 0

    time_convolve_strip = 0
    time_find_bounds = 0

    is_valid = False

    def __init__(self, strip_height, strip_idx, overall_dim) -> None:
        self.height, self.width = overall_dim
        self.RAD_PER_PX = self.DEPTH_HFOV / self.width

        self.idx = strip_idx
        self.strip_height = strip_height
        self.kernel = (strip_height, strip_height)
        self.search_min, self.search_max = 0, self.width
        self.bound_deadband = strip_height // 2 + 1

    def process(self, depth_img, previous_tracker):
        """Index is from bottom of image"""
        t = time.perf_counter()
        self.convolution, self.ymin, self.ymax, self.y_center = self.convolve_strip(
            depth_img
        )
        self.time_convolve_strip = time.perf_counter() - t

        t = time.perf_counter()
        self.left_bound, self.right_bound = self.find_bounds(self.convolution)
        self.x_center = self.left_bound + (self.right_bound - self.left_bound) // 2
        self.time_find_bounds = time.perf_counter() - t

        self.process_results(previous_tracker)
        self.is_valid = self.check_validity()

    def convolve_strip(self, depth_img):
        strip_bottom = self.height - (self.idx * self.strip_height)
        strip_top = strip_bottom - self.strip_height
        roi = depth_img[strip_top:strip_bottom]
        # print(self.height, self.strip_height, strip_top, strip_bottom, roi)

        # print(self.kernel)
        # convolution = convolve(roi, self.kernel, mode="valid")[0]
        # convolution = cv2.filter2D(src=roi, kernel=self.kernel, ddepth=-1, borderType=cv2.BORDER_ISOLATED)[0]
        # convolution = cv2.blur(src=roi, kernel=self.kernel, ddepth=-1, borderType=cv2.BORDER_ISOLATED)[0]
        # convolution = cv2.stackBlur(roi, self.kernel)
        convolution = cv2.blur(
            roi, (self.strip_height, self.strip_height), borderType=cv2.BORDER_ISOLATED
        )[0]
        # print(convolution)
        # FFT convolve the ROI with a box kernel to smooth depth signal
        return (
            convolution,
            strip_top,
            strip_bottom,
            (strip_top + (strip_bottom - strip_top) // 2),
        )

    def find_bounds(self, convolution, thresh_factor=0.2, search_start=None):
        search_start = search_start or len(convolution) // 2

        self.reference_distance = convolution[search_start]
        threshold = int(self.reference_distance * (1 - thresh_factor))

        l_bound = self.search_min + np.searchsorted(
            convolution[self.search_min : search_start], threshold, side="right"
        )

        # Reverse array before searching right bound -
        # searchsorted needs ascending vals
        r_bound = self.search_max - np.searchsorted(
            convolution[self.search_max : search_start : -1], threshold, side="right"
        )

        return l_bound, r_bound

    def process_results(self, previous_tracker):
        # Calculate detected width of the furrow
        rad = (self.right_bound - self.left_bound) * self.RAD_PER_PX
        self.furrow_width = abs(rad * self.reference_distance)

        if previous_tracker is not None:
            self.delta_x_from_previous = self.x_center - previous_tracker.x_center

    def check_validity(self):
        # Check if search found furrow wall
        if self.left_bound <= self.search_min + self.bound_deadband or self.right_bound >= self.search_max - self.bound_deadband:
            return False

        # Check that detected width is within bounds of an expected typical furrow
        if not self.FURROW_MIN_WIDTH <= self.furrow_width <= self.FURROW_MAX_WIDTH:
            return False

        return True

