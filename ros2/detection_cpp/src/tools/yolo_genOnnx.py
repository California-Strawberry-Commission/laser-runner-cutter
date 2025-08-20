from ultralytics import YOLO
import sys

if len(sys.argv) < 2:
  print("Usage: python yolo_genOnnx.py <model_path>")
  sys.exit(1)

model_path = sys.argv[1]
model = YOLO(model_path)
model.export(format="onnx", device=0, imgsz=(768, 1024), half=True, simplify=True)


