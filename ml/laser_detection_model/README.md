# Laser Detection ML Model

Project for preparing training data and training a ML model for detecting laser points in a color image.

## Environment setup

1.  Run `../runner_segmentation_model/scripts/env_setup.sh`.

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

1.  Source the venv

        $ source ../runner_segmentation_model/venv/bin/activate

1.  Obtain new training images. See `laser_detection/image_capture.py` for an example.

        $ python laser_detection/image_capture.py --output_directory <directory>

1.  Upload new images to the existing Labelbox dataset:

        $ python laser_detection/labelbox_api.py import_images --images_dir <directory>

1.  Annotate the images in Labelbox

1.  Obtain labels from Labelbox

    1.  Go to Annotate -> Laser Detection
    1.  Click the "Data Rows" tab
    1.  Click the "All (X data rows)" dropdown, then click "Export data"
    1.  Select all fields, then click the "Export JSON" button

1.  Create YOLO label txt files from the Labelbox ndjson export file:

        $ python laser_detection/labelbox_api.py create_labels --labelbox_export_file <ndjson export file path>

1.  Run `split_data.py` to split the raw image and label data into training and validation datasets. Be sure to remove the existing train/val images and labels beforehand.

        $ rm -rf data/prepared/images
        $ rm -rf data/prepared/labels
        $ python laser_detection/split_data.py

1.  Train the model locally:

        $ python laser_detection/yolov8_train.py

1.  Evaluate the trained model on the test dataset:

        $ python laser_detection/yolov8_eval.py <path to model>
