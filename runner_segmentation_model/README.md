# Runner Segmentation ML Model

Project for preparing training data and training a ML model for detecting instances of runners. Goal: given a color image, segment it into separate instances of runners.

## Environment setup

Note: the following steps are encapsulated in `scripts/env_setup.sh`.

1.  Install CUDA, a package that allows for model training on GPU's. This is all from https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

        $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        $ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        $ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
        $ sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
        $ sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
        $ sudo apt update
        $ sudo apt -y install cuda

1.  Create and source into a venv

    Install Python 3.11:

        $ sudo apt update && sudo apt upgrade -y
        $ sudo add-apt-repository -y ppa:deadsnakes/ppa
        $ sudo apt update
        $ sudo apt -y install python3.11 python3.11-venv python3.11-dev python3.11-tk

    Create venv:

        $ cd runner_segmentation_model
        $ python3.11 -m venv venv
        $ source venv/bin/activate

1.  Install specific version of PyTorch to match the CUDA version

        $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    NOTE: If you have any existing installs of touch torchvision or torchaudio in the venv, this will cause errors and they should be uninstalled.

1.  Install necessary requirements

        $ pip install -r requirements.txt

1.  Ensure opencv-python

        $ pip list | grep opencv

    If you see multiple versions (for example, both opencv-python and opencv-python-headless), you may need to reinstall opencv-python:

        $ pip uninstall opencv-python-headless -y
        $ pip uninstall opencv-python -y
        $ pip install opencv-python

### On Jetson

1.  Install CUDA, a package that allows for model training on GPU's. This is all from https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

        $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-ubuntu2004.pin
        $ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        $ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb
        $ sudo dpkg -i cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb
        $ sudo cp /var/cuda-tegra-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
        $ sudo apt-get update
        $ sudo apt-get -y install cuda

1.  Create and source into a venv

    Create venv:

        $ cd runner_segmentation_model
        $ python3.8 -m venv venv
        $ source venv/bin/activate

1.  Install specific version of PyTorch and torchvision. Follow instructions at https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048. Example:

    PyTorch:

        $ wget https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl ~/Downloads/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
        $ sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
        $ pip install 'Cython<3'
        $ pip install numpy ~/Downloads/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

    torchvision:

        $ sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
        $ git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
        $ cd torchvision
        $ export BUILD_VERSION=0.16.1
        $ python3 setup.py install

    Verify:

        $ python
        >>> import torch
        >>> print(torch.__version__)
        >>> print('CUDA available: ' + str(torch.cuda.is_available()))
        >>> print('cuDNN version: ' + str(torch.backends.cudnn.version()))
        >>> import torchvision
        >>> print(torchvision.__version__)

    NOTE: If you have any existing installs of PyTorch or torchvision in the venv, this will cause errors and they should be uninstalled.

1.  Install necessary requirements

        $ pip install -r requirements.txt

### DVC setup

DVC abstracts cloud storage and versioning of data for machine learning. It allows a directory structure to be pushed and pulled from an S3 bucket, so all developers can expect everyone to have the same project structure. DVC also has functionality for tracking results from different model runs, and building out experiments and pipelines.

1.  Get an AWS access key for the S3 bucket, and set it for use by DVC:

        $ dvc remote modify --local runner_segmentation access_key_id <access key>
        $ dvc remote modify --local runner_segmentation secret_access_key <secret access key>

1.  Pull data from the S3 bucket

        $ dvc pull -r runner_segmentation

## Workflow

1.  Split the raw image and label data into training and validation datasets. Be sure to remove the existing train/val/test images and labels beforehand.

        Remove existing split:
        $ rm -rf data/prepared
        Split the images first:
        $ python runner_segmentation/split_data.py images
        Split the YOLO labels to match the image split:
        $ python runner_segmentation/split_data.py yolo_labels
        Split the instanced masks (used by Mask R-CNN) to match the image split:
        $ python runner_segmentation/split_data.py masks

1.  Train and evaluate the YOLOv8 model locally:

        $ python runner_segmentation/yolov8.py train
        $ python runner_segmentation/yolov8.py eval --weights_file <path to trained weights>

1.  Train and evaluate the PyTorch Mask R-CNN model locally:

        $ python runner_segmentation/mask_rcnn.py train
        $ python runner_segmentation/mask_rcnn.py eval --weights_file <path to trained weights>

1.  Train and evaluate the Detectron2 Mask R-CNN model locally:

        $ python runner_segmentation/detectron.py train
        $ python runner_segmentation/detectron.py eval --weights_file <path to trained weights>
