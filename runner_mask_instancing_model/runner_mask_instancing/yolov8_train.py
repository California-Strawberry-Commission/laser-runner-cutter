"""File: yolov8_train.py

Description: Script to train a YOLOv8 model
"""

import os
from ultralytics import settings, YOLO

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
settings.update(
    {
        "datasets_dir": project_path,
        "runs_dir": os.path.join(project_path, "ultralytics/runs"),
        "weights_dir": os.path.join(project_path, "ultralytics/weights"),
    }
)

model = YOLO("yolov8n-seg.yaml")
results = model.train(
    data=os.path.join(project_path, "dataset.yml"),
    imgsz=(1024, 768),
    device=0,
    batch=-1,
    epochs=150,
    flipud=0.5,
)
