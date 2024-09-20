# Runner Segmentation ML Model

Project for preparing training data and training a ML model for detecting instances of runners. Goal: given a color image, segment it into separate instances of runners.

## Environment setup

1.  Run `scripts/env_setup.sh`.

1.  Install MMDetection

        $ pip install -U openmim
        $ mim install mmengine mmcv mmdet

1.  Check opencv-python

        $ pip list | grep opencv

    If you see multiple versions (for example, both opencv-python and opencv-python-headless), you may need to reinstall opencv-python:

        $ pip uninstall opencv-python-headless -y
        $ pip uninstall opencv-python -y
        $ pip install opencv-python

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
        $ python runner_segmentation/split_data.py images --input_dir data/raw/runner1800/images --output_dir data/prepared/runner1800/images
        Split the YOLO labels to match the image split:
        $ python runner_segmentation/split_data.py yolo_labels --input_dir data/raw/runner1800/labels --output_dir data/prepared/runner1800/labels
        Split the instanced masks (used by Mask R-CNN) to match the image split:
        $ python runner_segmentation/split_data.py masks --input_dir data/raw/runner1800/masks --output_dir data/prepared/runner1800/masks

1.  Train and evaluate the YOLOv8 model locally:

        $ python runner_segmentation/yolo.py train
        $ python runner_segmentation/yolo.py eval --weights_file <path to trained weights>

1.  Train and evaluate the PyTorch Mask R-CNN model locally:

        $ python runner_segmentation/mask_rcnn.py train
        $ python runner_segmentation/mask_rcnn.py eval --weights_file <path to trained weights>

1.  Train and evaluate a Detectron2 model locally:

        $ python runner_segmentation/detectron.py train
        $ python runner_segmentation/detectron.py eval --weights_file <path to trained weights>

1.  Train and evaluate a MMDetection model locally:

        $ python runner_segmentation/mmdetection.py train --model_name "mask-rcnn_r50"
        $ python runner_segmentation/mmdetection.py eval --model_name "mask-rcnn_r50" --weights_file <path to trained weights>
