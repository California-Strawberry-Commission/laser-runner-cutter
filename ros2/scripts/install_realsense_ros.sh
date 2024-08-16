#!/bin/bash
set -e

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/env.sh

# perform all commands in workspace dir
cd $ROS_WS_DIR

# Install RealSense SDK 2.0
# The following is from https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
sudo apt-get install -y apt-transport-https
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update

sudo apt-get install -y librealsense2-dev librealsense2-dbg

# Install ROS components
source /opt/ros/$ROS_DISTRO/local_setup.bash # Get ROS-specific env

echo "Installing ROS realsense"

# Install dependencies
sudo apt install -y "ros-$ROS_DISTRO-librealsense2*"
