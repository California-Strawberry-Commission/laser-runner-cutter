"""File: test_model.py

Description: Script to test a trained model
"""

import argparse
from ultralytics import YOLO


def main(model_path):
    model = YOLO(model_path)
    metrics = model.val(data="dataset.yml")
    print(f"mAP@75: {metrics.seg.map75}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained model")
    parser.add_argument(
        "model",
        help="Path to the trained model parameters file",
    )
    args = parser.parse_args()
    main(args.model)
