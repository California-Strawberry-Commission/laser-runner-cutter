import time
from typing import List
import numpy as np
import cv2
import time
from scipy.stats import linregress
from furrow_strip_tracker import FurrowStripTracker

MARKER_SIZE = 10
DISPLAY_LINES = True
BG_WIDTH = 140

class FurrowTracker:
    last_process_time = 0
    strips: List[FurrowStripTracker] = []
    
    width = None
    height = None
    
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
        
        depth_shape = depth_img.shape
        self.height, self.width = depth_shape
        self.strip_height = self.height // self.num_strips

        self.pin_y = 2 * self.height // 4
        
        for i in range(self.num_strips):
            self.strips.append(FurrowStripTracker(self.strip_height, i, depth_shape))
            
    def annotate(self, img, draw_timings=False):
        if not self.width and not self.height:
            return
        
        if draw_timings:
            cv2.rectangle(img, (0, 0), (BG_WIDTH, self.height), (127, 127, 0), -1)
        
        # Annotate strips
        for s in self.strips:
            s.annotate(img, draw_timings)

        # Annotate pin
        if pin_x := self.get_reg_x(self.pin_y):
            cv2.drawMarker(
                img,
                ( pin_x, self.pin_y),
                (0, 255, 0),
                markerType=cv2.MARKER_TRIANGLE_DOWN,
                markerSize=MARKER_SIZE * 2,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        
            # Annotate guidance
            x = self.width // 2 + self.guidance_offset_x
            cv2.line(img, (x, self.height), (x, self.pin_y), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.arrowedLine(img, (x, self.pin_y), (pin_x, self.pin_y), (0, 255, 255), 2, cv2.LINE_AA, tipLength = 0.5)

        
        cv2.putText(
            img,
            f"{self.last_process_time * 1000:.1f}ms",
            (0, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if self.reg_intercept is not None:
            y0, y1 = 0, self.height
            x0 = self.get_reg_x(y0) # (x0 - self.reg_intercept) / (self.reg_slope or 1)
            x1 = self.get_reg_x(y1)

            try:
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1, cv2.LINE_AA)
            except Exception:
                print("ERROR DRAWING LINE", x0, x1, y0, y1)
                raise Exception("sda")


            
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
        