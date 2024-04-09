""" File: create_yolo_labels.py

Description: Convert instanced mask images into YOLO segment labels
"""

import argparse
import os
import numpy as np
from PIL import Image
from . import segment_utils


def create_yolo_labels(masks_dir, output_dir, class_id=0):
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
            for file in files:
                file_path = os.path.join(subdir_path, file)
                mask_image = Image.open(file_path)
                mask_image_array = np.array(mask_image)
                segment = segment_utils.convert_mask_to_yolo_segment(mask_image_array)
                yolo_label_file.write(f"{class_id} {' '.join(map(str, segment))}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert instanced mask images into YOLO segment labels"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Path to the directory containing subdirs with instanced masks",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to where the YOLO label files will be written to",
    )
    args = parser.parse_args()
    create_yolo_labels(args.input_dir, args.output_dir)
