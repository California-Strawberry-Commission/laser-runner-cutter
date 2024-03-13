from dataclasses import dataclass

@dataclass
class Roi:
    xmin = 0
    xmax = -1
    ymin = 0
    ymax = -1
    

@dataclass
class FurrowTrackerAnnotationConfig:
    marker_size: int = 15
    display_strips = True
    
@dataclass
class FurrowTrackerConfig:
    annotation = FurrowTrackerAnnotationConfig()
    guidance_roi = Roi()

