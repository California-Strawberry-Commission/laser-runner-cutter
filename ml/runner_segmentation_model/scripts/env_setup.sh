#!/bin/bash

usage() { echo "Usage: $0 [-a <AWS access key used by DVC>] [-s <AWS secret key used by DVC>]" 1>&2; exit 1; }

aws_dvc_access_key=""
aws_dvc_secret_key=""
while getopts ":a:s:" opt; do
  case $opt in
    a)
      aws_dvc_access_key="$OPTARG"
      ;;
    s)
      aws_dvc_secret_key="$OPTARG"
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))
if [ -z "${aws_dvc_access_key}" ] || [ -z "${aws_dvc_secret_key}" ]; then
  usage
fi

# Install CUDA Toolkit 12.4, which allows for model training and inference on GPUs.
# The following is from https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# Install AWS CLI and zip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update
sudo apt -y install zip

# Install Python 3.11
sudo apt update && sudo apt upgrade -y
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt -y install python3.11
sudo apt -y install python3.11-venv
sudo apt -y install python3.11-dev

# Create and source into a venv
script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
cd "$script_dir/.."
python3.11 -m venv venv
source venv/bin/activate

# Update pip
pip install pip --upgrade

# Install specific version of PyTorch and torchvision to match the CUDA version.
# The following is from https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install necessary requirements
pip install -r requirements.txt

# DVC setup
dvc remote modify --local runner_segmentation access_key_id $aws_dvc_access_key
dvc remote modify --local runner_segmentation secret_access_key $aws_dvc_secret_key
dvc pull -r runner_segmentation
