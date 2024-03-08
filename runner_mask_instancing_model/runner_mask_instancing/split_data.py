"""File: split_data.py

Description: Script to split dataset between training and validation datasets
"""

import argparse
import os
import math
import shutil
import random
from glob import glob

RAW_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/raw",
)
IMAGES_DIR = os.path.join(RAW_DATA_DIR, "images")
LABELS_DIR = os.path.join(RAW_DATA_DIR, "labels")
PREPARED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/prepared",
)
TRAIN_IMAGES_DIR = os.path.join(PREPARED_DATA_DIR, "images/train")
TRAIN_LABELS_DIR = os.path.join(PREPARED_DATA_DIR, "labels/train")
VAL_IMAGES_DIR = os.path.join(PREPARED_DATA_DIR, "images/val")
VAL_LABELS_DIR = os.path.join(PREPARED_DATA_DIR, "labels/val")
TEST_IMAGES_DIR = os.path.join(PREPARED_DATA_DIR, "images/test")
TEST_LABELS_DIR = os.path.join(PREPARED_DATA_DIR, "labels/test")
VAL_RATIO = 0.20
TEST_RATIO = 0


def main(
    images_dir=IMAGES_DIR,
    labels_dir=LABELS_DIR,
    train_images_dir=TRAIN_IMAGES_DIR,
    train_labels_dir=TRAIN_LABELS_DIR,
    val_images_dir=VAL_IMAGES_DIR,
    val_labels_dir=VAL_LABELS_DIR,
    test_images_dir=TEST_IMAGES_DIR,
    test_labels_dir=TEST_LABELS_DIR,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
):
    if (
        os.path.exists(images_dir)
        and os.path.isdir(images_dir)
        and os.path.exists(labels_dir)
        and os.path.isdir(labels_dir)
    ):
        if not os.path.exists(train_images_dir):
            os.makedirs(train_images_dir)
        if not os.path.exists(train_labels_dir):
            os.makedirs(train_labels_dir)
        if not os.path.exists(val_images_dir):
            os.makedirs(val_images_dir)
        if not os.path.exists(val_labels_dir):
            os.makedirs(val_labels_dir)
        if not os.path.exists(test_images_dir):
            os.makedirs(test_images_dir)
        if not os.path.exists(test_labels_dir):
            os.makedirs(test_labels_dir)

        img_paths = glob(os.path.join(images_dir, "*.jpg")) + glob(
            os.path.join(images_dir, "*.png")
        )
        random.shuffle(img_paths)
        num_val = math.ceil(len(img_paths) * val_ratio)
        num_test = math.ceil(len(img_paths) * test_ratio)
        for index, img_path in enumerate(img_paths):
            _, img_filename = os.path.split(img_path)
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_filename)
            if not os.path.exists(label_path):
                continue

            if index < num_val:
                shutil.copy(img_path, os.path.join(val_images_dir, img_filename))
                shutil.copy(label_path, os.path.join(val_labels_dir, label_filename))
            elif index < num_val + num_test:
                shutil.copy(img_path, os.path.join(test_images_dir, img_filename))
                shutil.copy(label_path, os.path.join(test_labels_dir, label_filename))
            else:
                shutil.copy(
                    img_path,
                    os.path.join(train_images_dir, img_filename),
                )
                shutil.copy(
                    label_path,
                    os.path.join(train_labels_dir, label_filename),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset between training, validation, and test datasets"
    )
    parser.add_argument(
        "--images_dir",
        default=IMAGES_DIR,
    )
    parser.add_argument(
        "--labels_dir",
        default=LABELS_DIR,
    )
    parser.add_argument(
        "--train_images_dir",
        default=TRAIN_IMAGES_DIR,
    )
    parser.add_argument(
        "--train_labels_dir",
        default=TRAIN_LABELS_DIR,
    )
    parser.add_argument(
        "--val_images_dir",
        default=VAL_IMAGES_DIR,
    )
    parser.add_argument(
        "--val_labels_dir",
        default=VAL_LABELS_DIR,
    )
    parser.add_argument(
        "--test_images_dir",
        default=TEST_IMAGES_DIR,
    )
    parser.add_argument(
        "--test_labels_dir",
        default=TEST_LABELS_DIR,
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=VAL_RATIO,
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=TEST_RATIO,
    )

    args = parser.parse_args()
    main(
        args.images_dir,
        args.labels_dir,
        args.train_images_dir,
        args.train_labels_dir,
        args.val_images_dir,
        args.val_labels_dir,
        args.test_images_dir,
        args.test_labels_dir,
        args.val_ratio,
        args.test_ratio,
    )
