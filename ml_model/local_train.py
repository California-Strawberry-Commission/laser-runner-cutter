from ultralytics import YOLO

model = YOLO('yolov8n-seg.yaml')
results = model.train(data="runner-seg.yml", imgsz=(960, 720), device = 0, epochs=1000, flipud=.5)
