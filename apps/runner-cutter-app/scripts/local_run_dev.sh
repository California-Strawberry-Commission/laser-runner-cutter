#!/bin/bash
# For local development. Starts ROS2 nodes, Rosbridge, and the app concurrently.

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
app_dir=$script_dir/..
repo_dir=$script_dir/../../..

start_ros() {
  cd $repo_dir/ros2/scripts
  source setup.sh
  ros2 launch runner_cutter_control launch.py
}

start_rosbridge() {
  cd $app_dir
  ./scripts/run_rosbridge.sh
}

start_app() {
  cd $app_dir
  npm run dev
}

start_ros & start_rosbridge & start_app

wait
