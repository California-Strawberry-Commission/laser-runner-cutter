#!/bin/bash
set -e

script_dir="$(dirname "$(realpath "${BASH_SOURCE[-1]:-${(%):-%x}}")")"
source $script_dir/env.sh

# IMPORTANT: Some ROS versions need an upgrade 
# before install or they brick the system :)
sudo apt update
sudo apt upgrade -y

# ROS install
# Follow: https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install APT deps
sudo apt update
sudo apt install -y ros-foxy-desktop ros-foxy-diagnostic-updater python3-rosdep2 python3-colcon-common-extensions python3-argcomplete ros-foxy-rosbridge-suite
