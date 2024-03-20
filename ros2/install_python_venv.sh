#!/bin/bash

script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
source $script_dir/env.sh

if [ ! -d $VENV_DIR ]; then
    echo "ROS venv not created - creating"
    sudo apt update
    sudo apt install -y $PYTHON_DEPS
    python3 -m venv $VENV_DIR --system-site-packages --symlinks
fi