"""File: image_capture.py

Description: Script to split dataset between training and validation datasets
"""

import os
import math
import shutil
import random
from glob import glob

data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data_store/laser_detection",
)
in_img_dir = os.path.join(data_dir, "raw")
in_label_dir = os.path.join(data_dir, "raw_labels")
train_img_dir = os.path.join(data_dir, "images/train")
train_label_dir = os.path.join(data_dir, "labels/train")
val_img_dir = os.path.join(data_dir, "images/val")
val_label_dir = os.path.join(data_dir, "labels/val")
val_ratio = 0.1

if (
    os.path.exists(in_img_dir)
    and os.path.isdir(in_img_dir)
    and os.path.exists(in_label_dir)
    and os.path.isdir(in_label_dir)
):
    if not os.path.exists(train_img_dir):
        os.makedirs(train_img_dir)
    if not os.path.exists(train_label_dir):
        os.makedirs(train_label_dir)
    if not os.path.exists(val_img_dir):
        os.makedirs(val_img_dir)
    if not os.path.exists(val_label_dir):
        os.makedirs(val_label_dir)

    img_paths = glob(os.path.join(in_img_dir, "*.jpg")) + glob(
        os.path.join(in_img_dir, "*.png")
    )
    random.shuffle(img_paths)
    num_val = math.ceil(len(img_paths) * val_ratio)
    for index, img_path in enumerate(img_paths):
        _, img_filename = os.path.split(img_path)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(in_label_dir, label_filename)
        if not os.path.exists(label_path):
            continue

        if index < num_val:
            shutil.copy(img_path, os.path.join(val_img_dir, img_filename))
            shutil.copy(label_path, os.path.join(val_label_dir, label_filename))
        else:
            shutil.copy(
                img_path,
                os.path.join(train_img_dir, img_filename),
            )
            shutil.copy(
                label_path,
                os.path.join(train_label_dir, label_filename),
            )
