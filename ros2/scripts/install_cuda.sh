#!/bin/bash
set -e

script_dir="$(dirname "$(realpath "${BASH_SOURCE[-1]:-${(%):-%x}}")")"
source $script_dir/env.sh
source $VENV_DIR/bin/activate

cd ~
arch=$(uname -i)
if [[ $arch == x86_64* ]]; then
    # Install CUDA Toolkit 12.4, which allows for model training and inference on GPUs.
    # The following is from https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4

    # Install specific version of PyTorch and torchvision to match the CUDA version.
    # The following is from https://pytorch.org/get-started/locally/
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
elif [[ $arch == aarch64* ]]; then
    # Install CUDA Toolkit 12.4, which allows for model training and inference on GPUs.
    # The following is from https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-tegra-repo-ubuntu2204-12-4-local_12.4.0-1_arm64.deb
    sudo dpkg -i cuda-tegra-repo-ubuntu2204-12-4-local_12.4.0-1_arm64.deb
    sudo cp /var/cuda-tegra-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4 cuda-compat-12-4

    # Install cuDNN
    # The following is from https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
    wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb
    sudo dpkg -i cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb
    sudo cp /var/cudnn-local-tegra-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cudnn-cuda-12

    # Install specific version of PyTorch and torchvision to match the CUDA version.
    # The following is based on https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
    wget https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl
    wget https://nvidia.box.com/shared/static/u0ziu01c0kyji4zz3gxam79181nebylf.whl -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
    wget https://nvidia.box.com/shared/static/9si945yrzesspmg9up4ys380lqxjylc3.whl -O torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
    pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl

    # Append exports to ~/.bashrc if it doesn't already exist
    lines_to_append=(
        'export PATH=/usr/local/cuda-12.4/bin:$PATH'
        'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH'
    )
    for line in "${lines_to_append[@]}"; do
        if ! grep -qxF "$line" ~/.bashrc; then
            echo "$line" >> ~/.bashrc
        fi
    done
fi