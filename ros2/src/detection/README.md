# Detection

This node subscribes to a ROS2 Image topic, runs inference (if enabled), and publishes detection results to a topic. Runner detection is done using TensorRT.

## Setup

1.  Build and install OpenCV with CUDA support:

        $ detection/scripts/build_opencv_cuda.sh

2.  If you haven't already, be sure to pull the git lfs files:

        $ git lfs pull

3.  Build the TensorRT Engine file(s) from the ONNX model file(s) in the `models` dir:

        $ detection/scripts/build_tensorrt_engines.sh
