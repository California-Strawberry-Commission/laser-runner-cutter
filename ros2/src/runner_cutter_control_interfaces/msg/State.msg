# Whether a camera-laser calibration model exists.
bool calibrated
# State machine state
string state
# The rect (min x, min y, width, height) representing the reach of the laser, normalized to the camera frame size
common_interfaces/Vector4 normalized_laser_bounds
# Number of camera-laser point correspondences collected for calibration
uint32 num_calibration_points
# Range of camera-space depth (mm) covered by the point correspondences used in
# the calibration model fit. 0 if there is no model.
common_interfaces/Vector2 calibration_depth_range
float32 calibration_mean_position_error