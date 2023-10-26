from ultralytics import YOLO

model = YOLO("yolov8n-pose.yaml")
results = model.train(
    data="dataset.yml",
    imgsz=(640, 480),
    device=0,
    batch=16,
    epochs=50,
    flipud=0.5,
)
