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
    # Ubuntu 22.04 uses Python 3.10, with a setuptools version where direct calls to setup.py has been deprecated.
    # setuptools 58.2.0 is the last version that works with ROS2 packages without warning.
    # See https://answers.ros.org/question/396439/setuptoolsdeprecationwarning-setuppy-install-is-deprecated-use-build-and-pip-and-other-standards-based-tools/
    $VENV_DIR/bin/python -m pip install setuptools==58.2.0
fi
