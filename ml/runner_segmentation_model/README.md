# Runner Segmentation ML Model

Project for preparing training data and training a ML model for detecting instances of runners. Goal: given a color image, segment it into separate instances of runners.

## Environment setup

1.  Run the setup script

        $ ./scripts/env_setup.sh

1.  [Optional] Install MMDetection

        $ pip install -U openmim
        $ mim install mmengine mmcv mmdet

1.  Check opencv-python

        $ pip list | grep opencv

    If you see multiple versions (for example, both opencv-python and opencv-python-headless), you may need to reinstall opencv-python:

        $ pip uninstall opencv-python-headless -y
        $ pip uninstall opencv-python -y
        $ pip install opencv-python

### DVC setup

We use DVC to manage training datasets and trained models. DVC abstracts cloud storage and versioning of data for machine learning. It allows a directory structure to be pushed and pulled from an S3 bucket, so all developers can expect everyone to have the same project structure. DVC also has functionality for tracking results from different model runs, and building out experiments and pipelines.

1.  Get an AWS access key and secret key for the S3 bucket used by DVC

1.  Run the DVC setup script

        $ ./scripts/dvc_setup.sh -a <AWS access key used by DVC> -s <AWS secret key used by DVC>

## Creating and labeling a new dataset

Labelbox is used for dataset annotation. `labelbox_api.py` provides convenience methods to upload images and extract annotations from Labelbox. Follow these steps in order to create a new dataset of runner images and to annotate the images using Labelbox.

1.  Source the venv

        $ source venv/bin/activate

1.  Get an API key from the Labelbox workspace, and add it to the `.env` file

        $ cd runner_segmentation_model
        $ echo "LABELBOX_API_KEY=<api key>" > .env

1.  Add the new runner images to `data/raw/<dataset name>/images`

1.  Upload the images to Labelbox

        $ python -m runner_segmentation.labelbox_api import_images -n <Labelbox Dataset Name> -i <path/to/dataset/images/from/previous/step>

1.  Annotate the images in Labelbox

    1.  On Labelbox, Go to Annotate -> New Project -> Image, and follow the steps there to create a new annotation project
    1.  It may be beneficial to use the latest runner detection model to provide predictions

            $ python -m runner_segmentation.labelbox_api upload_predictions -i <path/to/dataset/images/from/previous/step> -m <path/to/model/weights/best.pt -n <Labelbox Dataset Name>

1.  Obtain labels from Labelbox

    1. Go to Annotate -> `<Your Annotation Project Name>`
    1. Click the "Data Rows" tab
    1. Click the "All (X data rows)" dropdown, then click "Export data"
    1. Select all fields, then click the "Export JSON" button

1.  Create mask files from Labelbox annotations

        $ python -m runner_segmentation.labelbox_api create_masks -i <path/to/export.ndjson> -o data/raw/<dataset name>/masks -n <Labelbox Annotation Project Name>

1.  Create YOLO label files from mask files

        $ python -m ml_utils.create_yolo_labels -i data/raw/<dataset name>/masks -o data/raw/<dataset name>/labels

## Training

1.  Source the venv

        $ source venv/bin/activate

1.  Split the raw image and label data into training and validation datasets. Be sure to remove the existing train/val/test images and labels beforehand if it already exists.

    1.  Remove existing split:

            $ rm -rf data/prepared/<dataset name>

    1.  Split the images first:

            $ python -m runner_segmentation.split_data images --input_dir data/raw/<dataset name>/images --output_dir data/prepared/<dataset name>/images

    1.  Split the YOLO labels to match the image split:

            $ python -m runner_segmentation.split_data yolo_labels --input_dir data/raw/<dataset name>/labels --output_dir data/prepared/<dataset name>/labels

    1.  [Optional] Split the instanced masks (used by Mask R-CNN) to match the image split:

            $ python -m runner_segmentation.split_data masks --input_dir data/raw/<dataset name>/masks --output_dir data/prepared/<dataset name>/masks

1.  Train and evaluate a YOLOv8 model locally:

    1.  Modify `dataset.yml` with the desired dataset path

    1.  Run the training script

            $ python -m runner_segmentation.yolo train
            $ python -m runner_segmentation.yolo eval --weights_file path/to/trained/weights.pt

1.  Train and evaluate a PyTorch Mask R-CNN model locally:

        $ python -m runner_segmentation.mask_rcnn train --images_dir data/prepared/<dataset name>/images --masks_dir data/prepared/<dataset name>/masks
        $ python -m runner_segmentation.mask_rcnn eval --weights_file path/to/trained/weights.pt

1.  Train and evaluate a Detectron2 model locally:

        $ python -m runner_segmentation.detectron train --images_dir data/prepared/<dataset name>/images --masks_dir data/prepared/<dataset name>/masks
        $ python -m runner_segmentation.detectron eval --weights_file <path to trained weights>

1.  Train and evaluate a MMDetection model locally:

        $ python -m runner_segmentation.mmdetection train --model_name "mask-rcnn_r50" --data_dir data/prepared/<dataset name>
        $ python -m runner_segmentation.mmdetection eval --model_name "mask-rcnn_r50" --weights_file <path to trained weights>
