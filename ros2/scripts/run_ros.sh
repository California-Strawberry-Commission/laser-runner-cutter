#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

start_lifecycle_manager() {
  cd $script_dir
  source setup.sh
  ros2 launch lifecycle_manager launch.py
}

start_rosbridge() {
  cd $script_dir
  source setup.sh
  ros2 launch runner_cutter_control rosbridge_websocket_launch.xml
}

start_web_video_server() {
  cd $script_dir
  source setup.sh
  ros2 launch runner_cutter_control web_video_server_launch.py
}

start_runner_cutter() {
  cd $script_dir
  source setup.sh
  ros2 launch runner_cutter_control launch.py
}

start_lifecycle_manager & start_rosbridge & start_web_video_server & start_runner_cutter

wait
