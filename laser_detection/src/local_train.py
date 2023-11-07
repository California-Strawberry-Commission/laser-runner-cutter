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

model = YOLO("yolov8n.yaml")
results = model.train(
    data="dataset.yml",
    imgsz=(640, 480),
    device=0,
    batch=-1,
    epochs=100,
)
