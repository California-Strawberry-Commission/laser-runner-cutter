#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

start_ros() {
  cd $script_dir
  source setup.sh
  ros2 launch runner_cutter_control launch.py
}

start_ros_nav() {
  cd $script_dir
  source setup.sh
  ros2 launch guidance_brain component_launch.py
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

start_ros_nav & start_ros & start_rosbridge & start_web_video_server

wait
