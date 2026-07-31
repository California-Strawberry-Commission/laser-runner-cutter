# Type of detected object
uint8 detection_type  # DetectionType.msg
# Seconds since epoch
float64 timestamp
# Detected object instances
ObjectInstance[] instances
# 2D pixel coordinates on camera color frame that did not have valid 3D positions
common_interfaces/Vector2[] invalid_points
# Median pixel displacement (dx, dy) between the previous and current frame.
# Zero if not available (e.g. first frame).
common_interfaces/Vector2 displacement