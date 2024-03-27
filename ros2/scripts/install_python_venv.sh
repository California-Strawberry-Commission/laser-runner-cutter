#!/bin/bash
set -e

script_dir="$(dirname "$(realpath "${BASH_SOURCE[-1]:-${(%):-%x}}")")"
source $script_dir/env.sh

if [ ! -d $VENV_DIR ]; then
    echo "ROS venv not created - creating"
    sudo apt update
    sudo apt install -y python3-venv python3-pip
    python3 -m venv $VENV_DIR --system-site-packages --symlinks 

    # Update pip - old pip will run into install problems.
    $VENV_DIR/bin/python -m pip install --upgrade pip setuptools
fi
