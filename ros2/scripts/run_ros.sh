#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pids=()

# Run LiveKit
(
  cd "$script_dir/.."
  exec bash src/livekit_ros2/scripts/run.sh
) &
pids+=("$!")

ros_launch() {
  (
    cd "$script_dir"
    source setup.sh
    exec ros2 launch $1
  ) &
  pids+=("$!")
}

# Run ROS nodes
ros_launch "lifecycle_manager launch.py"
ros_launch "runner_cutter_control rosbridge_websocket_launch.xml"
ros_launch "runner_cutter_control launch.py launch_nav_nodes:=False"

# On Ctrl+C, forward SIGINT to all
trap '
  echo "[INFO] Stopping..."
  for pid in "${pids[@]}"; do
    kill -INT "$pid" 2>/dev/null || true
  done
  wait
' INT TERM

wait
