#!/bin/bash

APT_DEPS="ros-foxy-desktop python3-argcomplete python3-pip python3-rosdep2"

# IMPORTANT: Some ROS versions need an upgrade 
# before install or they brick the system :)
sudo apt update
sudo apt upgrade -y

# setup ROS to be installed if not already
if [ -z "$ROS_DISTRO" ]; then
    # Follow: https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html
    echo "!!!!!!!!!!! ROS is not installed - installing FOXY !!!!!!!!!!!"
    sudo apt install software-properties-common
    sudo add-apt-repository universe

    sudo apt update && sudo apt install curl -y
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    sudo apt update
    source /opt/ros/foxy/local_setup.bash # Get updated env vars so $ROS_DISTRO is populated for next installs
fi

sudo apt update
sudo apt install -y $APT_DEPS

# Install realsense-ros. Use GH install b/c need to build from source
(
    echo "Installing ROS realsense"

    # Install dependancies
    sudo apt install -y "ros-$ROS_DISTRO-librealsense2*"
    
    # cp realsense ROS packages into source dir
    cd "$script_dir/.."
    git submodule update --init --recursive
    cp -r realsense-ros/realsense2_* "$script_dir"

    # Setup for ROS
    sudo rosdep init --include-eol-distros      # "sudo rosdep init --include-eol-distros" for Foxy and earlier
    rosdep update --include-eol-distros         # "sudo rosdep update --include-eol-distros" for Foxy and earlier
    rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y
)


# Install python deps of subpackages
# (Don't bother using ROS's dep management for py)
script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )

# Find all requirement.txt files and iterate through them
find "$script_dir" -name 'requirements.txt' -type f | while read -r file; do
    # Extract directory path of the requirements.txt file
    dir_path=$(dirname "$file")

    # Navigate to the directory containing the requirements.txt file
    echo "Installing requirements from $file"
    pushd "$dir_path" || exit
    pip install -r "$(basename "$file")"
    popd || exit
done