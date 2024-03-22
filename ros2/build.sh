#!/bin/bash

script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
cd $script_dir

source $VENV_DIR/bin/activate
source /opt/ros/foxy/setup.sh

# Build
colcon build --symlink-install