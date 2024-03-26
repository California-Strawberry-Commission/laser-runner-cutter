#!/bin/bash
script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
source $script_dir/env.sh

source $VENV_DIR/bin/activate
source /opt/ros/foxy/setup.sh

# Build
cd $WS_DIR
colcon build --symlink-install