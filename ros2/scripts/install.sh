#!/bin/bash
set -e

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/env.sh

cd $script_dir

# Install - cache if already installed.
installed_file=".installed"
if [ ! -f "$installed_file" ]; then
    echo "Not installed - installing"

    # Init submodules
    git submodule update --init --recursive

    # Install Git LFS and pull LFS files
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get -y install git-lfs
    git lfs pull

    bash ./set_jetson_power_mode.sh
    bash ./install_python_venv.sh
    bash ./install_cuda.sh
    bash ./install_arena_sdk.sh
    bash ./create_heliosdac_udev.sh
    bash ./install_ros.sh
    bash ./install_realsense_ros.sh
    bash ./install_requirements.sh

    echo "Building..."
    bash ./build.sh

    touch "$installed_file"
fi
