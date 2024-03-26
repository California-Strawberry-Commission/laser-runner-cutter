#!/bin/bash
set -e

script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
source $script_dir/env.sh

cd $script_dir

# Install - cache if already installed.
if [ ! -f "$INSTALLED_F" ]; then
    echo "Not installed - installing"
    bash ./install_python_venv.sh
    bash ./install_ros.sh
    bash ./install_realsense_ros.sh
    bash ./install_requirements.sh

    echo "Building!"
    bash ./build.sh

    touch "$INSTALLED_F"
fi