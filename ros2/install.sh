#!/bin/bash

script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )

cd $script_dir

# Install - cache if already installed.
if [ ! -f "./installed" ]; then
    echo "Not installed - installing"
    sudo bash install_ros.sh
    sudo bash install_realsense_ros.sh
    sudo bash install_requirements.sh
    touch installed
fi