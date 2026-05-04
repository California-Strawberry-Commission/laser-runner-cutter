#!/bin/bash
set -eo pipefail

# Compile TensorRT engines if not already built.
# Engines are device-specific and can't be baked into the image.
echo "[entrypoint] Building TensorRT engines if needed..."
bash /workspaces/ros2/src/detection/scripts/build_tensorrt_engines.sh

# Build ROS workspace
source /opt/ros/${ROS_DISTRO}/setup.sh
echo "[entrypoint] Building ROS workspace..."
cd /workspaces/ros2
colcon build --symlink-install

# Launch ROS nodes
source /workspaces/ros2/install/setup.sh
echo "[entrypoint] Launching nodes..."
# Monitor mode (set -m) gives each backgrounded command its own process group, while 
# sharing the session and controlling terminal.
# Each ros2 launch process needs to be in its own process group in the background so that
# we can intercept SIGINT/SIGTERM signals in this script and forward the signals to each
# launch process, in order to control the shutdown sequence.
# In addition, we need each ros2 launch process to behave interactively so that the
# shutdown event loops run and thus log messages appear during shutdown.
set -m
# PIDs of all the launch processes
pids=()
ros_launch() {
    ros2 launch $1 &
    pids+=("$!")
}
ros_launch "lifecycle_manager launch.py"
ros_launch "livekit_ros2 launch.py"
ros_launch "runner_cutter_control rosbridge_websocket_launch.xml"
ros_launch "runner_cutter_control launch.py"

echo "[entrypoint] All launch processes started. PIDs: ${pids[*]}"

shutdown() {
    local sig=$1
    echo "[entrypoint] Caught signal $sig, forwarding to launch processes..."
    # Block further SIGINT/SIGTERM signals to prevent shutdown from being called
    # recursively
    trap '' "$sig"

    # Send signal to the launch processes. The launch processes will then disable node
    # respawns and handle shutting down its nodes in order.
    for pid in "${pids[@]}"; do
        kill -"$sig" "$pid" 2>/dev/null || true
    done

    # Wait up to 10 seconds for graceful shutdown, polling each second
    local deadline=$((SECONDS + 10))
    while [ $SECONDS -lt $deadline ]; do
        local any_running=false
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && any_running=true && break
        done
        $any_running || break
        sleep 1
    done

    # Force kill any process groups that are still running
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && kill -9 -"$pid" 2>/dev/null || true
    done

    wait 2>/dev/null || true
    echo "[entrypoint] Shutdown complete."
}

# Trigger shutdown on SIGINT or SIGTERM
trap 'shutdown INT' SIGINT
trap 'shutdown TERM' SIGTERM

# Block the script indefinitely until background jobs are stopped
wait || true

echo "[entrypoint] All processes stopped."
