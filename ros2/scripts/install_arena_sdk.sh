#!/bin/bash
set -e

script_dir="$(dirname "$(realpath "${BASH_SOURCE[-1]:-${(%):-%x}}")")"
source $script_dir/env.sh
source $VENV_DIR/bin/activate

# Install Arena SDK
cd ~
arch=$(uname -i)
if [[ $arch == x86_64* ]]; then
    curl -L -o ArenaSDK_v0.1.90_Linux_x64.tar.gz "https://drive.google.com/uc?export=download&id=11ZAeLbj6xQulD_qYQdQ1ee44UZyh_zBk"
    mkdir ~/ArenaSDK
    tar -xvzf ArenaSDK_v0.1.90_Linux_x64.tar.gz -C ~/ArenaSDK
    cd ~/ArenaSDK/ArenaSDK_Linux_x64
    sudo sh Arena_SDK_Linux_x64.conf
elif  [[ $arch == aarch64* ]]; then
    curl -L -o ArenaSDK_v0.1.68_Linux_ARM64.tar.gz "https://drive.google.com/uc?export=download&id=1hcwyD8UfgDPkv0p4bt9zRH3-AlKMFbwk"
    mkdir ~/ArenaSDK
    tar -xvzf ArenaSDK_v0.1.68_Linux_ARM64.tar.gz -C ~/ArenaSDK
    cd ~/ArenaSDK/ArenaSDK_Linux_ARM64
    sudo sh Arena_SDK_ARM64.conf
fi

# Install Arena Python Package
cd ~
curl -L -o arena_api-2.5.9-py3-none-any.whl "https://drive.google.com/uc?export=download&id=1B8stioWii3-aQAMTIepXkVxR-WKlWYI5"
pip install arena_api-2.5.9-py3-none-any.whl