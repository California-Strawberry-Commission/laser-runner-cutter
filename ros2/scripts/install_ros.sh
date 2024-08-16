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
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install APT deps
sudo apt update
sudo apt install -y ros-$ROS_DISTRO-desktop ros-dev-tools ros-$ROS_DISTRO-rosbridge-suite ros-$ROS_DISTRO-async-web-server-cpp

# https://github.com/matplotlib/matplotlib/issues/26827#issuecomment-1726026699
# Solves a potential import conflict between system matplotlib and env matplotlib
sudo apt remove -y python3-matplotlib