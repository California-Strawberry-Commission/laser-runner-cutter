#!/bin/bash
set -e

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Init submodules
git submodule update --init --recursive

# Install Git LFS and pull LFS files
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get -y install git-lfs
git lfs pull

# Set power mode (supported Jetson devices only)
bash ./set_jetson_power_mode.sh

# Install Docker
bash ./install_docker.sh

# Create udev rule for Helios DAC
bash ../src/laser_control/scripts/create_heliosdac_udev.sh

# Create .env file with generated LiveKit API secret
bash ./create_env_file.sh

# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh
