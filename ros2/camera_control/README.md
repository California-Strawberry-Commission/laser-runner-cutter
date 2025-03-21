# Camera Control

ROS2 node and client for capturing camera frames and running runner and laser detection.

## Models

ML models in the `models/` directory are stored in Git LFS, and can be pulled via `git lfs pull`. The `.pt` file is the trained YOLOv8 model. The TensorRT engine files (`.engine` extension) were built using:

        $ yolo export model=camera_control/models/RunnerSegYoloV8l.pt format=engine imgsz=768,1024 half=True simplify=True device=0
