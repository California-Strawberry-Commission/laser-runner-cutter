#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

start_ros() {
  cd $script_dir
  source setup.sh
  ros2 launch runner_cutter_control launch.py
}

start_rosbridge() {
  cd $script_dir
  source setup.sh
  ros2 launch rosbridge_server rosbridge_websocket_launch.xml
}

start_ros & start_rosbridge

wait
