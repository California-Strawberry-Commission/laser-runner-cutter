# Machine Learning

Directory withing the laser runner removal project for preparing data and training ML models

## Environment setup

1.  Install CUDA, this is a package that allows for model training on GPU's. This is all from https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

        $wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        $sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        $wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
        $sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
        $sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
        $sudo apt-get update
        $sudo apt-get -y install cuda

1.  Create and source into a venv

        $ cd ml_model
        $ python3 -m venv venv
        $ source venv/bin/activate

1.  Install a specific version of pytorch to match cuda versions and detect GPUs

        $ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    NOTE: If you have any existing installs of touch torchvision or torchaudio in the venv, this will cause errors and they should be uninstalled

1.  Install necessary requirements

        $ pip install -r requirements.txt

## DVC setup

DVC is a project that abstracts cloud storage of large data for machine learning. It allows a directory structure to be push and pull from a s3 bucket, so all developers can expect everyone to have the same folder structure. DVC also has functionality for tracking results from different model runs, and building out experiments and pipelines.

1.  Get an AWS access key for the S3 bucket, and set it for use by DVC:

        $ dvc remote modify --local segmentation_data access_key_id <access key>
        $ dvc remote modify --local segmentation_data secret_access_key <secret access key>

1.  Pull data from the S3 bucket

        $ dvc pull -r segmentation_data

### Labelbox integration

Labelbox is used for dataset annotation. `labelbox_api.py` provides convenience methods to upload images and extract annotations from Labelbox.

1.  Get an API key from the Labelbox workspace, and add it to the `.env` file.

        $ cd ml_model
        $ echo "LABELBOX_API_KEY=<api key>" > .env

## Workflow

1.  Obtain new training images

1.  Use `import_images` from `labelbox_api.py` to upload new images to the existing Labelbox dataset

1.  Annotate the images in Labelbox

1.  Obtain labels from Labelbox

    1. Go to Annotate -> Laser Detection
    1. Click the "Data Rows" tab
    1. Click the "All (X data rows)" dropdown, then click "Export data v2"
    1. Select all fields, then click the "Export JSON" button

1.  Use `create_yolo_labels_from_segment_ndjson` in `labelbox_api.py` to create YOLO label txt files from the Labelbox ndjson export file

1.  Run `split_data.py` to split the raw image and label data into training and validation datasets. Be sure to remove the existing train/val images and labels beforehand.

        $ rm -rf data_store/segmentation_data/images
        $ rm -rf data_store/segmentation_data/labels
        $ python split_data.py

1.  The local train script uses the Ultralytics repo to train a laser detection model on the dataset:

        $ python local_train.py

1.  Test the trained model

        $ python test_model.py <path to model>
