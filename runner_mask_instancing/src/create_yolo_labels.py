""" File: create_yolo_labels.py

Description: Convert instanced mask images into YOLO segment labels
"""

import argparse
import os
import numpy as np
from PIL import Image
from segment_utils import convert_mask_to_yolo_segment

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/raw",
)
MASKS_DIR = os.path.join(DATA_DIR, "masks")
OUT_DIR = os.path.join(DATA_DIR, "labels")


def create_yolo_labels(masks_dir=MASKS_DIR, output_dir=OUT_DIR, class_id=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subdirs = [
        d for d in os.listdir(masks_dir) if os.path.isdir(os.path.join(masks_dir, d))
    ]
    for subdir in subdirs:
        yolo_label_name = subdir + ".txt"
        yolo_label_path = os.path.join(output_dir, yolo_label_name)
        with open(yolo_label_path, "w") as yolo_label_file:
            subdir_path = os.path.join(masks_dir, subdir)
            files = [
                f
                for f in os.listdir(subdir_path)
                if os.path.isfile(os.path.join(subdir_path, f))
            ]
            yolo_label_file.write(f"{class_id}")
            for file in files:
                file_path = os.path.join(subdir_path, file)
                mask_image = Image.open(file_path)
                mask_image_array = np.array(mask_image)
                segment = convert_mask_to_yolo_segment(mask_image_array)
                yolo_label_file.write(" ".join(map(str, segment)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert instanced mask images into YOLO segment labels"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        default=MASKS_DIR,
        help="Path to the directory containing the instanced masks",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=OUT_DIR,
        help="Path to where the YOLO label files will be written to",
    )
    args = parser.parse_args()
    create_yolo_labels(args.input_dir, args.output_dir)
