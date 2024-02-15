"""File: yolov8_eval.py

Description: Script to evaluate a trained YOLOv8 model
"""

import argparse
import os
from ultralytics import YOLO

PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main(model_path):
    model = YOLO(model_path)
    metrics = model.val(
        data=os.path.join(PROJECT_PATH, "dataset.yml"),
        split="test",
    )
    print(f"mAP50: {metrics.seg.map50}")
    print(f"mAP50-95: {metrics.seg.map}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained YOLOv8 model")
    parser.add_argument(
        "model",
        help="Path to the trained model parameters file",
    )
    args = parser.parse_args()
    main(args.model)
