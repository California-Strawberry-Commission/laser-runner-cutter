#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
app_dir=$script_dir/..
repo_dir=$script_dir/../../..

cd $repo_dir/ros2/scripts
source setup.sh
ros2 launch rosbridge_server rosbridge_websocket_launch.xml