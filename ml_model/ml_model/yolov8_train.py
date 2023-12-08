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

model = YOLO("yolov8n-seg.yaml")
results = model.train(
    data="ml_model/runner-seg.yml", imgsz=(960, 720), device=0, epochs=100, flipud=0.5
)
model.val()
