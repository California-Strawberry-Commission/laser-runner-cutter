#!/bin/bash
set -e

source env.sh
cd $SCRIPT_DIR

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