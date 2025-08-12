from ultralytics import YOLO
import sys

# Load a trained YOLOv8 PyTorch model
if len(sys.argv) < 2:
  print("Usage: python yolo_reformat.py <model_path>")
  sys.exit(1)

model_path = sys.argv[1]
model = YOLO(model_path)

# Export the model to TensorRT engine format
# 'device=0' specifies GPU device, 'half=True' enables FP16 quantization for faster inference
model.export(format="engine", device=0) 