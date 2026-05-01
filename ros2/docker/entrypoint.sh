#!/bin/bash
set -e

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
pids=()
ros_launch() {
    # setsid puts each launch process in its own process group, so we can
    # signal the whole tree (launch process + all spawned ROS nodes) at once.
    setsid ros2 launch $1 &
    pids+=("$!")
}
ros_launch "lifecycle_manager launch.py"
ros_launch "runner_cutter_control rosbridge_websocket_launch.xml"
ros_launch "runner_cutter_control launch.py launch_nav_nodes:=False"

shutdown() {
    echo "[entrypoint] Stopping..."

    # SIGINT to each process group. Tells ros2 launch and all its nodes to shut down gracefully.
    for pid in "${pids[@]}"; do
        kill -INT -"$pid" 2>/dev/null || true
    done

    # Wait up to 10 seconds for graceful shutdown
    local deadline=$((SECONDS + 10))
    while [ $SECONDS -lt $deadline ]; do
        local any_running=false
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && any_running=true && break
        done
        $any_running || break
        sleep 1
    done

    # Force kill any process groups that didn't exit in time
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && kill -9 -"$pid" 2>/dev/null || true
    done

    wait 2>/dev/null || true
    echo "[entrypoint] Stopped."
}

trap shutdown INT TERM

wait || true
