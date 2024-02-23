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

# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update

# Install Python 3.11
sudo apt update && sudo apt upgrade -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
sudo apt install python3.11-venv

# Create and source into a venv
script_dir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
cd "$script_dir/.."
python3.11 -m venv venv
source venv/bin/activate

# Install specific version of PyTorch to match the CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install necessary requirements
pip install -r requirements.txt

# DVC setup
dvc remote modify --local runner_segmentation access_key_id $aws_dvc_access_key
dvc remote modify --local runner_segmentation secret_access_key $aws_dvc_secret_key
dvc pull -r runner_segmentation
