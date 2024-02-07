# Laser Detection ML Model

Project for preparing training data and training a ML model for detecting laser points in a color image.

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

        $ cd laser_detection_model
        $ python3 -m venv venv
        $ source venv/bin/activate

1.  Install a specific version of pytorch to match cuda versions and detect GPUs

        $ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    NOTE: If you have any existing installs of touch torchvision or torchaudio in the venv, this will cause errors and they should be uninstalled

1.  Install necessary requirements

        $ pip install -r requirements.txt

### DVC setup

DVC abstracts cloud storage and versioning of data for machine learning. It allows a directory structure to be pushed and pulled from an S3 bucket, so all developers can expect everyone to have the same project structure. DVC also has functionality for tracking results from different model runs, and building out experiments and pipelines.

1.  Get an AWS access key for the S3 bucket, and set it for use by DVC:

        $ dvc remote modify --local laser_detection access_key_id <access key>
        $ dvc remote modify --local laser_detection secret_access_key <secret access key>

1.  Pull data from the S3 bucket

        $ dvc pull -r laser_detection

### Labelbox integration

Labelbox is used for dataset annotation. `labelbox_api.py` provides convenience methods to upload images and extract annotations from Labelbox.

1.  Get an API key from the Labelbox workspace, and add it to the `.env` file.

        $ cd laser_detection_model
        $ echo "LABELBOX_API_KEY=<api key>" > .env

## Workflow

1.  Obtain new training images. See `src/image_capture.py` for an example.

        $ python src/image_capture --output_directory <directory>

1.  Upload new images to the existing Labelbox dataset:

        $ python src/labelbox_api import_images --images_dir <directory>

1.  Annotate the images in Labelbox

1.  Obtain labels from Labelbox

    1. Go to Annotate -> Laser Detection
    1. Click the "Data Rows" tab
    1. Click the "All (X data rows)" dropdown, then click "Export data v2"
    1. Select all fields, then click the "Export JSON" button

1.  Create YOLO label txt files from the Labelbox ndjson export file:

        $ python src/labelbox_api create_labels --labelbox_export_file <ndjson export file path>

1.  Run `split_data.py` to split the raw image and label data into training and validation datasets. Be sure to remove the existing train/val images and labels beforehand.

        $ rm -rf data/prepared/images
        $ rm -rf data/prepared/labels
        $ python src/split_data.py

1.  Train the model locally:

        $ python src/yolov8_train.py

1.  Evaluate the trained model on the test dataset:

        $ python src/yolov8_eval.py <path to model>
