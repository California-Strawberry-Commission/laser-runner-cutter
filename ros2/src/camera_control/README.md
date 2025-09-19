# Camera Control

ROS2 node and client for capturing camera frames and running runner and laser detection.

## Models

ML models in the `models/` directory are stored in Git LFS, and can be pulled via `git lfs pull`. The `.pt` file is the trained YOLOv8 model. TensorRT engine files (`.engine` extension) can be built using:

    $ ./scripts/build_tensorrt_models.sh

The TensorRT engine files will be created in the `models/` directory.
