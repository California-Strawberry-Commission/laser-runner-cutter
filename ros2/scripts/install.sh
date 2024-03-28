#!/bin/bash
set -e

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/env.sh

cd $script_dir

# Install - cache if already installed.
installed_file=".installed"
if [ ! -f "$installed_file" ]; then
    echo "Not installed - installing"
    bash ./install_python_venv.sh
    bash ./install_ros.sh
    bash ./install_realsense_ros.sh
    bash ./install_requirements.sh

    echo "Building!"
    bash ./build.sh

    touch "$installed_file"
fi