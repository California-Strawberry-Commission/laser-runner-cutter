# https://unix.stackexchange.com/questions/4650/how-to-determine-the-path-to-a-sourced-tcsh-or-bash-shell-script-from-within-the/692485#692485
FULL_PATH_TO_SCRIPT="$(realpath "${BASH_SOURCE[-1]}")"
SCRIPT_DIR="$(dirname "$FULL_PATH_TO_SCRIPT")"

ROS_DEPS="ros-foxy-desktop ros-foxy-diagnostic-updater python3-rosdep2 python3-colcon-common-extensions python3-argcomplete"
PYTHON_DEPS="python3-venv python3-pip"
VENV_DIR=~/.ros_venv
WS_DIR=$SCRIPT_DIR/..
INSTALLED_F=".installed"