#!/bin/bash
# For local development. Starts ROS2 nodes, Rosbridge, and the app concurrently.

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
app_dir=$script_dir/..
repo_dir=$app_dir/../..
ros_dir=$repo_dir/ros2

start_ros() {
  cd $ros_dir
  ./scripts/run_ros.sh
}

start_app() {
  cd $app_dir
  npm run dev
}

start_ros & start_app

wait
