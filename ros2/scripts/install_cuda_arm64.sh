#!/bin/bash
set -e

script_dir="$(dirname "$(realpath "${BASH_SOURCE[-1]:-${(%):-%x}}")")"
source $script_dir/env.sh
source $VENV_DIR/bin/activate

cd ~

# Install CUDA, a package that allows for model training and inference on GPUs.
# This is from https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=20.04&target_type=deb_local
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb
sudo dpkg -i cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb
sudo cp /var/cuda-tegra-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install specific versions of PyTorch and torchvision.
# Based on https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# PyTorch:
sudo apt-get -y install python3-pip libopenblas-base libopenmpi-dev libomp-dev libcudnn8
wget https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
pip install 'Cython<3' numpy torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
# torchvision:
sudo apt-get -y install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.16.1
python setup.py install

# Append exports to ~/.bashrc if it doesn't already exist
lines_to_append=(
    'export PATH=/usr/local/cuda-11.8/bin:$PATH'
    'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH'
    'export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1'
)
for line in "${lines_to_append[@]}"; do
    if ! grep -qxF "$line" ~/.bashrc; then
        echo "$line" >> ~/.bashrc
    fi
done