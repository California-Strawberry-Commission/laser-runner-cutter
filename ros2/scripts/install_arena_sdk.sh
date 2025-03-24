#!/bin/bash
set -e

script_dir="$(dirname "$(realpath "${BASH_SOURCE[-1]:-${(%):-%x}}")")"
source $script_dir/env.sh
source $VENV_DIR/bin/activate

# Install gdown, which is needed to download large files from Google Drive
pip install gdown

# Install Arena SDK
cd ~
arch=$(uname -i)
if [[ $arch == x86_64* ]]; then
    gdown "https://drive.google.com/uc?id=1Vs0P9zoY8mvOaWPVK2tYuako1K6i9TZo" --output ArenaSDK_v0.1.91_Linux_x64.tar.gz
    sudo mkdir /opt/ArenaSDK
    sudo tar -xvzf ArenaSDK_v0.1.91_Linux_x64.tar.gz -C /opt/ArenaSDK
    cd /opt/ArenaSDK/ArenaSDK_Linux_x64
    sudo sh Arena_SDK_Linux_x64.conf
elif  [[ $arch == aarch64* ]]; then
    gdown "https://drive.google.com/uc?id=1cD2GqK06rWDhDDq9EsD6j4qIF3UeHVXE" --output ArenaSDK_v0.1.73_Linux_ARM64.tar.gz
    sudo mkdir /opt/ArenaSDK
    sudo tar -xvzf ArenaSDK_v0.1.73_Linux_ARM64.tar.gz -C /opt/ArenaSDK
    cd /opt/ArenaSDK/ArenaSDK_Linux_ARM64
    sudo sh Arena_SDK_ARM64.conf
fi

# Install Arena Python Package
cd ~
gdown "https://drive.google.com/uc?id=1B8stioWii3-aQAMTIepXkVxR-WKlWYI5" --output arena_api-2.5.9-py3-none-any.whl
pip install arena_api-2.5.9-py3-none-any.whl