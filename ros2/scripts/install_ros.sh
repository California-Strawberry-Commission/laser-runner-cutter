#!/bin/bash
set -e

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/env.sh

# IMPORTANT: Some ROS versions need an upgrade before install or they brick the system :)
# See https://github.com/ros2/ros2/issues/1272
sudo apt update
sudo apt upgrade -y

# Install ROS2 Humble
# The following is from https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
sudo apt install -y software-properties-common
sudo add-apt-repository universe -y

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

# Install APT deps
sudo apt update
sudo apt install -y ros-$ROS_DISTRO-desktop ros-dev-tools ros-$ROS_DISTRO-diagnostic-updater ros-$ROS_DISTRO-rosbridge-suite ros-$ROS_DISTRO-async-web-server-cpp

# https://github.com/matplotlib/matplotlib/issues/26827#issuecomment-1726026699
# Solves a potential import conflict between system matplotlib and env matplotlib
sudo apt remove -y python3-matplotlib

# libusb is required by some of our ROS packages
# Note: Ubuntu 22.04 already has this installed
sudo apt install -y libusb-1.0-0-dev

# opencv is required by some of our ROS packages
sudo apt install -y libopencv-dev