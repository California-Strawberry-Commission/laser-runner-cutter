#! /bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
app_dir=$script_dir/..
repo_dir=$script_dir/../../..

start_ros() {
  cd $repo_dir/ros2/scripts
  source setup.sh
  ros2 launch runner_cutter_control launch.py
}

start_rosbridge() {
  cd $repo_dir/ros2/scripts
  source setup.sh
  ros2 launch rosbridge_server rosbridge_websocket_launch.xml
}

start_app() {
  cd $app_dir
  npm run dev
}

start_ros & start_rosbridge & start_app

wait
