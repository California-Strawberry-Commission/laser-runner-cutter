# Median pixel displacement (dx, dy) between the previous and current frame.
common_interfaces/Vector2 pixel_displacement
# 3D displacement (relative to camera) corresponding to `pixel_displacement`,
# derived by converting the pixel displacement to positions via depth frame.
# Zero if not available (e.g. depth/alignment unavailable).
common_interfaces/Vector3 position_displacement
# Time elapsed, in seconds, between the previous and current frame used to
# compute `pixel_displacement`.
float64 delta_time_secs
