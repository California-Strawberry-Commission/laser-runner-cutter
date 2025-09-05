#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Start Docker Compose detached so Ctrl+C won't be absorbed
( cd "$script_dir/.." && docker compose up -d )

# Wait for LiveKit Ingress server to be ready
sleep 2

pids=()
ros_launch() {
  (
    cd "$script_dir/../../scripts"
    source setup.sh
    exec ros2 launch $1
  ) &
  pids+=("$!")
}

ros_launch "livekit_ros2 launch.py"

# On Ctrl+C, forward SIGINT to ros2 launch, then bring Docker Compose down
trap '
  echo "[INFO] Stopping..."
  for pid in "${pids[@]}"; do
    kill -INT "$pid" 2>/dev/null || true
  done
  wait
  ( cd "$script_dir/.." && docker compose down -t 5 ) || true
' INT TERM

wait
