""" File: create_coco_label.py

Description: Convert instanced mask images into COCO format JSON file
"""

import argparse
import os
import numpy as np
from glob import glob
from natsort import natsorted
import cv2
import json
from segment_utils import convert_mask_to_rle
from tqdm import tqdm


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def create_coco_labels(images_dir, masks_dir, output_filepath):
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    images = []
    annotations = []
    obj_count = 0
    img_paths = glob(os.path.join(images_dir, "*.jpg")) + glob(
        os.path.join(images_dir, "*.png")
    )
    img_paths = natsorted(img_paths)
    for idx, img_path in enumerate(tqdm(img_paths)):
        img_filename = os.path.basename(img_path)
        height, width = cv2.imread(img_path).shape[:2]
        images.append(
            {"id": idx, "file_name": img_filename, "width": width, "height": height}
        )

        mask_subdir_name = os.path.splitext(img_filename)[0]
        mask_subdir_path = os.path.join(masks_dir, mask_subdir_name)
        mask_filenames = [
            f
            for f in os.listdir(mask_subdir_path)
            if os.path.isfile(os.path.join(mask_subdir_path, f))
        ]
        for mask_file in mask_filenames:
            mask_filepath = os.path.join(mask_subdir_path, mask_file)
            mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            # Don't include masks without any valid pixel values
            if np.max(mask) < 255:
                continue

            # Calculate the bounding box
            nonzero_indices = np.argwhere(mask > 0)
            min_x = np.min(nonzero_indices[:, 1])
            max_x = np.max(nonzero_indices[:, 1])
            min_y = np.min(nonzero_indices[:, 0])
            max_y = np.max(nonzero_indices[:, 0])
            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

            # Calculate the area
            area = np.sum(mask > 0)

            rle_counts, rle_size = convert_mask_to_rle(mask)
            annotation = {
                "image_id": idx,
                "id": obj_count,
                "category_id": 0,
                "bbox": bbox,
                "area": area,
                "segmentation": {"size": rle_size, "counts": rle_counts},
                "iscrowd": 0,
            }
            annotations.append(annotation)
            obj_count += 1

    coco_format_json = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "runner"}],
    }
    with open(output_filepath, "w") as output_file:
        json.dump(coco_format_json, output_file, cls=NpEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert instanced mask images into COCO format JSON file"
    )

    parser.add_argument(
        "-i",
        "--images_dir",
        required=True,
        help="Path to the directory containing images",
    )
    parser.add_argument(
        "-m",
        "--masks_dir",
        required=True,
        help="Path to the directory containing subdirs with instanced masks for the images in images_dir",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="Output COCO label JSON file path",
    )
    args = parser.parse_args()
    create_coco_labels(args.images_dir, args.masks_dir, args.output_file)
