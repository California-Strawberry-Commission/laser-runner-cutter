#!/bin/bash
set -e

# The following is from https://docs.docker.com/engine/install/ubuntu/
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
# Install the Docker packages
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