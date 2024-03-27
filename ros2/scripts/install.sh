#!/bin/bash
set -e

script_dir="$(dirname "$(realpath "${BASH_SOURCE[-1]:-${(%):-%x}}")")"
source $script_dir/env.sh

cd $script_dir

# Install - cache if already installed.
installed_file=".installed"
if [ ! -f "$installed_file" ]; then
    echo "Not installed - installing"
    bash ./install_python_venv.sh

    arch=$(uname -i)
    if [[ $arch == x86_64* ]]; then
        bash ./install_cuda_amd64.sh
    elif  [[ $arch == aarch64* ]]; then
        bash ./install_cuda_arm64.sh
    fi

    bash ./install_ros.sh
    bash ./install_realsense_ros.sh
    bash ./install_requirements.sh

    echo "Building!"
    bash ./build.sh

    touch "$installed_file"
fi