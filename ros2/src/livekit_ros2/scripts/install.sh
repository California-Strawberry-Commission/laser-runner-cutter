#!/bin/bash
set -e

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install GStreamer
# The following is from https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

# Install Docker
# The following is from https://docs.docker.com/engine/install/ubuntu/
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Note: on Jetson, you may run into an issue with Docker Compose port mapping. If so, downgrade Docker.
# See https://forums.developer.nvidia.com/t/issue-with-docker-compose-port-mapping-on-jetson-orin-nano-iptable-raw-module-missing/335728/4 for more details on the issue.
if ! sudo iptables -t raw -L -n &>/dev/null; then
    sudo apt install -y docker-ce=5:27.5* docker-ce-cli=5:27.5* --allow-downgrades
fi

# Add user to the Docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Python deps
source $script_dir/../../../scripts/env.sh
source $VENV_DIR/bin/activate
pip install -r $script_dir/../requirements.txt