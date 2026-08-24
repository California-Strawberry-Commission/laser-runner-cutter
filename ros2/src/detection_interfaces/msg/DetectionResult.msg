# Type of detected object
uint8 detection_type  # DetectionType.msg
builtin_interfaces/Time timestamp
# Detected object instances
ObjectInstance[] instances
# 2D pixel coordinates on camera color frame that did not have valid 3D positions
common_interfaces/Vector2[] invalid_points
# Frame-to-frame optical flow displacement. Zero (default-constructed) if not
# available (e.g. first frame).
FlowDisplacement flow_displacement