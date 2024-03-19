#!/bin/bash

script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )

cd $script_dir

bash install.sh

# Set up ROS workspace
source /opt/ros/foxy/local_setup.bash
source install/local_setup.bash

# Build
colcon build --symlink-install