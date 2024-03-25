#!/bin/bash
source env.sh

source $VENV_DIR/bin/activate
source /opt/ros/foxy/setup.sh

# Build
cd $WS_DIR
colcon build --symlink-install