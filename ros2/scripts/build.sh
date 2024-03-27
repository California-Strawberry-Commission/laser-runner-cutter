#!/bin/bash
script_dir="$(dirname "$(realpath "${BASH_SOURCE[-1]:-${(%):-%x}}")")"
source $script_dir/env.sh

source $VENV_DIR/bin/activate
source /opt/ros/foxy/setup.sh

# Build
cd $ROS_WS_DIR
colcon build --symlink-install