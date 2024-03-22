from dataclasses import dataclass

import numpy as np

@dataclass
class Roi:
    xmin = 0
    xmax = -1
    ymin = 0
    ymax = -1
    
@dataclass
class CameraConfig:
    horizontal_depth_fov = np.deg2rad(87)  # Realsense - 87 degrees
    centerline_offset_x = 0
    
@dataclass
class AnnotationConfig:
    marker_size: int = 15
    display_strips = True
    
@dataclass
class FurrowTrackerConfig:
    annotation = AnnotationConfig()
    camera = CameraConfig()
    guidance_roi = Roi()
    
    furrow_min_width = 10 * 25.4  # 10 inches
    furrow_max_width = 20 * 25.4  # 10 inches

    

