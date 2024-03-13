#!/bin/bash

script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )

sudo apt update
sudo apt install python3-pip

# Install realsense-ros
(
    echo "Installing ROS realsense"

    # Install dependancies
    sudo apt install "ros-$ROS_DISTRO-librealsense2*"
    
    # cp realsense ROS packages into source dir
    cd "$script_dir/.."
    git submodule update --init --recursive
    cp -r realsense-ros/realsense2_* "$script_dir"

    # Install more dependancies
    sudo apt-get install python3-rosdep -y
    sudo rosdep init # "sudo rosdep init --include-eol-distros" for Foxy and earlier
    rosdep update # "sudo rosdep update --include-eol-distros" for Foxy and earlier
    rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y
)

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