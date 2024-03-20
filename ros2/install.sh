#!/bin/bash
set -e

script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
cd $script_dir

# Install - cache if already installed.
if [ ! -f "installed" ]; then
    echo "Not installed - installing"
    bash ./install_python_venv.sh
    bash ./install_ros.sh
    bash ./install_realsense_ros.sh
    bash ./install_requirements.sh
    touch installed

    echo "Building!"
    bash ./build.sh
fi