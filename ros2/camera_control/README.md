# Camera Control

ROS2 node and client for capturing camera frames and running runner and laser detection.

## Setup

ML model weights are stored in S3 via DVC. DVC abstracts cloud storage and versioning of data for machine learning. It allows a directory structure to be pushed and pulled from an S3 bucket, so all developers can expect everyone to have the same project structure.

1.  Get an AWS access key for the S3 bucket, and set it for use by DVC:

        $ dvc remote modify --local deployed_models access_key_id <access key>
        $ dvc remote modify --local deployed_models secret_access_key <secret access key>

1.  Pull data from the S3 bucket

        $ dvc pull -r deployed_models
