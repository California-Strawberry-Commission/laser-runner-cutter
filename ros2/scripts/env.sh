SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[-1]:-${(%):-%x}}")")"
ROS_DEPS="ros-foxy-desktop ros-foxy-diagnostic-updater python3-rosdep2 python3-colcon-common-extensions python3-argcomplete"
PYTHON_DEPS="python3-venv python3-pip"
VENV_DIR=~/.ros_venv
WS_DIR=$SCRIPT_DIR/..
