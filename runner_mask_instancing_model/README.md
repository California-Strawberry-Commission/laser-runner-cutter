# Runner Mask Instancing ML Model

Project for preparing training data and training a ML model for instancing mask images. Goal: given a binary mask that represents runners, segment it into separate instances of runners.

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

    Install Python 3.11:

        $ sudo apt update && sudo apt upgrade -y
        $ sudo add-apt-repository ppa:deadsnakes/ppa
        $ sudo apt update
        $ sudo apt install python3.11
        $ sudo apt install python3.11-venv

    Create venv:

        $ cd runner_segmentation_model
        $ python3.11 -m venv venv
        $ source venv/bin/activate

1.  Install specific version of PyTorch to match the CUDA version

        $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    NOTE: If you have any existing installs of touch torchvision or torchaudio in the venv, this will cause errors and they should be uninstalled

1.  Install necessary requirements

        $ pip install -r requirements.txt

### DVC setup

DVC abstracts cloud storage and versioning of data for machine learning. It allows a directory structure to be pushed and pulled from an S3 bucket, so all developers can expect everyone to have the same project structure. DVC also has functionality for tracking results from different model runs, and building out experiments and pipelines.

1.  Get an AWS access key for the S3 bucket, and set it for use by DVC:

        $ dvc remote modify --local runner_mask_instancing access_key_id <access key>
        $ dvc remote modify --local runner_mask_instancing secret_access_key <secret access key>

1.  Pull data from the S3 bucket

        $ dvc pull -r runner_mask_instancing

### Labelbox integration

Labelbox is used for dataset annotation. `labelbox_api.py` provides convenience methods to upload images and extract annotations from Labelbox.

1.  Get an API key from the Labelbox workspace, and add it to the `.env` file.

        $ cd runner_mask_instancing_model
        $ echo "LABELBOX_API_KEY=<api key>" > .env

## Workflow

1.  Upload non-instanced runner mask images to the existing Labelbox dataset:

        $ python src/labelbox_api import_images --images_dir <path to images dir>

1.  Annotate the images in Labelbox with one polyline for each runner instance

1.  Obtain labels from Labelbox

    1. Go to Annotate -> Runner Mask Instancing
    1. Click the "Data Rows" tab
    1. Click the "All (X data rows)" dropdown, then click "Export data v2"
    1. Select all fields, then click the "Export JSON" button

1.  Using the Labelbox polyline annotations, instance the mask images

        $ python src/instance_mask_images.py -f <path to ndjson export>

1.  Create YOLO label txt files from the instanced masks:

        $ python src/create_yolo_labels.py

1.  Run `split_data.py` to split the raw image and label data into training and validation datasets. Be sure to remove the existing train/val images and labels beforehand.

        $ rm -rf data/prepared/images
        $ rm -rf data/prepared/labels
        $ python src/split_data.py

1.  Train the model locally:

        $ python src/yolov8_train.py

1.  Evaluate the trained model on the test dataset:

        $ python src/yolov8_eval.py <path to model>
