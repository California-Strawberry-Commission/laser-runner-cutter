#!/bin/bash
set -e

script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
source $script_dir/env.sh

# https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages
sudo apt-get install -y apt-transport-https

# Install realsense SDK
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update

sudo apt-get install -y librealsense2-dev librealsense2-dbg

# Install ROS components
source /opt/ros/foxy/local_setup.bash # Get ROS-specific env

echo "Installing ROS realsense"

# Install dependancies
sudo apt install -y "ros-$ROS_DISTRO-librealsense2*"

# Init realsense submodule
git submodule update --init --recursive

# Setup for ROS
# sudo rosdep init --include-eol-distros      # "sudo rosdep init --include-eol-distros" for Foxy and earlier
# rosdep update --include-eol-distros         # "sudo rosdep update --include-eol-distros" for Foxy and earlier