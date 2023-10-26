"""File: image_capture.py

Description: Script to split dataset between training and validation datasets
"""

import os
import math
import shutil

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

    img_files = [
        f
        for f in sorted(os.listdir(in_img_dir), key=lambda x: x.lower())
        if os.path.isfile(os.path.join(in_img_dir, f))
    ]
    num_val = math.ceil(len(img_files) * val_ratio)
    val_files = img_files[:: math.floor(len(img_files) / num_val)][:num_val]
    print(val_files)
    for img_filename in img_files:
        img_filepath = os.path.join(in_img_dir, img_filename)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_filepath = os.path.join(in_label_dir, label_filename)
        if not os.path.exists(label_filepath):
            continue

        if img_filename in val_files:
            shutil.copy(img_filepath, os.path.join(val_img_dir, img_filename))
            shutil.copy(label_filepath, os.path.join(val_label_dir, label_filename))
        else:
            shutil.copy(
                img_filepath,
                os.path.join(train_img_dir, img_filename),
            )
            shutil.copy(
                label_filepath,
                os.path.join(train_label_dir, label_filename),
            )
