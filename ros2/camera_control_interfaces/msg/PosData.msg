# Seconds since epoch
float64 timestamp
# 3D positions relative to camera
common_interfaces/Vector3[] pos_list
# 2D pixel coordinates on camera color frame with associated valid 3D position
common_interfaces/Vector2[] point_list
# 2D pixel coordinates on camera color frame without valid 3D position
common_interfaces/Vector2[] invalid_point_list