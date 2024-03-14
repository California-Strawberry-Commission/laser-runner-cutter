#!/bin/bash
script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )

APT_DEPS="ros-foxy-desktop python3-argcomplete python3-pip python3-rosdep2 python3-colcon-common-extensions ros-foxy-diagnostic-updater"

# IMPORTANT: Some ROS versions need an upgrade 
# before install or they brick the system :)
sudo apt update
sudo apt upgrade -y

# ROS install
# Follow: https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html
echo "!!!!!!!!!!! ROS is not installed - installing FOXY !!!!!!!!!!!"
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
source /opt/ros/foxy/local_setup.bash # Get updated env vars so $ROS_DISTRO is populated for next installs


# Install APT deps
sudo apt update
sudo apt install -y $APT_DEPS

