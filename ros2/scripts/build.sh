#!/bin/bash
script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/env.sh

source $VENV_DIR/bin/activate
source /opt/ros/$ROS_DISTRO/setup.sh

# Build
cd $ROS_WS_DIR
colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
