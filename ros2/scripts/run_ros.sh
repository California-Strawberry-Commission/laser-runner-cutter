#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

start_ros() {
  cd $script_dir
  source setup.sh
  ros2 launch runner_cutter_control launch.py
}

start_ros

wait
