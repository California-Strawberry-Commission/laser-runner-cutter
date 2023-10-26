import os
from ultralytics import settings, YOLO

project_path = os.path.dirname(os.path.abspath(__file__))
settings.update(
    {
        "datasets_dir": project_path,
        "runs_dir": os.path.join(project_path, "ultralytics/runs"),
        "weights_dir": os.path.join(project_path, "ultralytics/weights"),
    }
)

model = YOLO("yolov8n-pose.yaml")
results = model.train(
    data="dataset.yml",
    imgsz=(640, 480),
    device=0,
    batch=16,
    epochs=50,
    flipud=0.5,
)
