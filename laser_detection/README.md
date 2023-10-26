# Laser Detection ML Model

## Environment setup

1.  Install CUDA, this is a package that allows for model training on GPU's. This is all from https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

        $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        $ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        $ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
        $ sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
        $ sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
        $ sudo apt-get update
        $ sudo apt-get -y install cuda

1.  Create and source into a venv

        $ cd laser_detection
        $ python3 -m venv venv
        $ source venv/bin/activate

1.  Install a specific version of pytorch to match cuda versions and detect GPUs

        $ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    NOTE: If you have any existing installs of touch torchvision or torchaudio in the venv, this will cause errors and they should be uninstalled

1.  Install necessary requirements

        $ pip install -r requirements.txt

## DVC setup

DVC abstracts cloud storage and versioning of data for machine learning. It allows a directory structure to be pushed and pulled from an S3 bucket, so all developers can expect everyone to have the same project structure. DVC also has functionality for tracking results from different model runs, and building out experiments and pipelines.

1.  Get an AWS access key for the S3 bucket, and set it for use by DVC:

        $ dvc remote modify --local laser_detection access_key_id <access key>
        $ dvc remote modify --local laser_detection secret_access_key <secret access key>

## Labelbox integration

Labelbox is used for dataset annotation. `labelbox_api.py` provides convenience methods to upload images and extract annotations from Labelbox.

1.  Get an API key from the Labelbox workspace, and add it to the `.env` file.

        $ cd laser_detection
        $ echo "LABELBOX_API_KEY=<api key>" > .env

## Update Ultralytics directory files

The ultralytics repo uses a config file to read in some directory locations, changing those to point to the project directory makes things a bit cleaner and easier.

1.  Update the directory paths in `~/.config/Ultralytics/settings.yml` to:

        datasets_dir: /home/<user_name>/ros_ws/src/LaserRunnerRemoval/laser_detection/
        weights_dir: /home/<user_name>/ros_ws/src/LaserRunnerRemoval/laser_detection/ultralytics/weights
        runs_dir: /home/<user_name>/ros_ws/src/LaserRunnerRemoval/laser_detection/ultralytics/runs

## Train the model

1.  The local train script uses the ultalytics repo to train a laser detection model on the dataset:

        $ python local_train.py
