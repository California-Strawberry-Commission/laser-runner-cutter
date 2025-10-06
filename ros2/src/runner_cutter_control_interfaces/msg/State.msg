bool calibrated
# State machine state
string state
# The rect (min x, min y, width, height) representing the reach of the laser, normalized to the camera frame size
common_interfaces/Vector4 normalized_laser_bounds