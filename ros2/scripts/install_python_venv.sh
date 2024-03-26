#!/bin/bash
set -e

source env.sh

if [ ! -d $VENV_DIR ]; then
    echo "ROS venv not created - creating"
    sudo apt update
    sudo apt install -y $PYTHON_DEPS
    python3 -m venv $VENV_DIR --system-site-packages --symlinks 

    # Update pip - old pip will run into install problems.
    $VENV_DIR/bin/python -m pip install --upgrade pip
fi
