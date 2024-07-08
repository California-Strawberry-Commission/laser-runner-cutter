#!/bin/bash
set -e

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/env.sh

if [ ! -d $VENV_DIR ]; then
    echo "ROS venv not created - creating"
    sudo apt update
    sudo apt install -y python3-venv python3-pip
    python3 -m venv $VENV_DIR --system-site-packages --symlinks 

    # Update pip - old pip will run into install problems.
    $VENV_DIR/bin/python -m pip install --upgrade pip
    # Latest setuptools (v70) doesn't work with PyTorch, so downgrade to a specific version...
    $VENV_DIR/bin/python -m pip install setuptools==68.2.0
fi
