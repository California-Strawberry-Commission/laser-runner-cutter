#!/bin/bash
script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/env.sh

source $VENV_DIR/bin/activate
source /opt/ros/$ROS_DISTRO/setup.sh

# Build TensorRT models
bash $script_dir/../camera_control/scripts/build_tensorrt_models.sh

# Fresh build of ROS2 nodes
cd $ROS_WS_DIR
rm -rf build install log
colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
