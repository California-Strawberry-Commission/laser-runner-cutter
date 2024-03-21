# Laser Runner Cutter

Laser Runner Cutter is a project by the California Strawberry Commission for automated detection and cutting of strawberry runners (also known as stolons).

## Packages

### ROS2/

- `runner_cutter_control`: node for laser runner cutter control and automation.
- `runner_cutter_control_interfaces`: custom interface definitions used in `runner_cutter_control`.
- `laser_control`: node and client for controlling laser DACs. Currently supports Helios and Ether Dream DACs.
- `laser_control_interfaces`: custom interface definitions used in `laser_control`.
- `camera_control`: node and client for capturing camera frames and running runner and laser detection. Currently supports RealSense cameras.
- `camera_control_interfaces`: custom interface definitions used in `camera_control`.
- `common_interfaces`: shared interface definitions
- `rqt_laser_runner_cutter`: rqt plugin for visualization and monitoring of relevant ROS2 topics.

### ML/

- `runner_segmentation_model`: ML pipeline and models for instance segmentation of runners
- `laser_detection_model`: ML pipeline and models for object detection of lasers
- `runner_mask_instancing_model`: ML pipeline and models for instace segmentation of semantic segmented runner masks. Goal: given a single binary mask that represents runners, segment it into separate instances of runners.

### Apps/

- `runner-cutter-app`: Web app for laser runner cutter control and automation.
- `runner_image_capture_amiga_kivy`: An application deployable to the Amiga brain, written using Kivy, for capturing runner images in the field.
- `runner_image_capture_amiga_react`: An application deployable to the Amiga brain, written using React and FastAPI, for capturing runner images in the field.

### Tools/

- `simulator`: Unity-based simulator.

## Setup and run

1.  Setup and run ROS2 nodes: see `ros2/README.md`

1.  Setup and run Runner Cutter App: see `apps/runner-cutter-app/README.md`
