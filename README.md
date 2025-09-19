# Laser Runner Cutter

Laser Runner Cutter is a project by the California Strawberry Commission for automated detection and cutting of strawberry runners (also known as stolons).

## Packages

### ros2/

- `amiga_control`: node for controlling the Farm-ng Amiga.
- `amiga_control_interfaces`: custom interface definitions used in `amiga_control`.
- `camera_control`: node and client for capturing camera frames and running runner and laser detection. Currently supports RealSense cameras.
- `camera_control_interfaces`: custom interface definitions used in `camera_control`.
- `common_interfaces`: shared interface definitions.
- `furrow_perceiver`: node for detection of furrows, used for navigation.
- `furrow_perceiver_interfaces`: custom interface definitions used in `furrow_perceiver`.
- `guidance_brain`: node for navigation automation.
- `guidance_brain_interfaces`: custom interface definitions used in `guidance_brain`.
- `laser_control`: node and client for controlling laser DACs. Currently supports Helios and Ether Dream DACs.
- `laser_control_interfaces`: custom interface definitions used in `laser_control`.
- `lifecycle_manager`: node to manage node lifecycles.
- `runner_cutter_control`: node for laser runner cutter control and automation.
- `runner_cutter_control_interfaces`: custom interface definitions used in `runner_cutter_control`.

### ml/

- `laser_detection_model`: ML pipeline and models for object detection of lasers
- `ml_utils`: shared logic used by the other ML projects
- `runner_mask_instancing_model`: ML pipeline and models for instace segmentation of semantic segmented runner masks. Goal: given a single binary mask that represents runners, segment it into separate instances of runners.
- `runner_segmentation_model`: ML pipeline and models for instance segmentation of runners

### apps/

- `runner-cutter-app`: web app for laser runner cutter control and automation.

## Setup and run

1.  Setup and run ROS2 nodes: see `ros2/README.md`

1.  Setup and run Runner Cutter App: see `apps/runner-cutter-app/README.md`
