"""File: split_data.py

Description: Script to split dataset between training, validation, and test datasets.
"""

import argparse
import os
import math
import shutil
import random
from glob import glob

TRAIN_DIRNAME = "train"
VAL_DIRNAME = "val"
TEST_DIRNAME = "test"

DEFAULT_RAW_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/raw/runner1800",
)
DEFAULT_PREPARED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/prepared/runner1800",
)
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


def split_images(
    input_dir, output_dir, val_ratio=DEFAULT_VAL_RATIO, test_ratio=DEFAULT_TEST_RATIO
):
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        return

    train_dir = os.path.join(output_dir, TRAIN_DIRNAME)
    val_dir = os.path.join(output_dir, VAL_DIRNAME)
    test_dir = os.path.join(output_dir, TEST_DIRNAME)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    img_paths = glob(os.path.join(input_dir, "*.jpg")) + glob(
        os.path.join(input_dir, "*.png")
    )
    random.shuffle(img_paths)
    num_val = math.ceil(len(img_paths) * val_ratio)
    num_test = math.ceil(len(img_paths) * test_ratio)
    for index, img_path in enumerate(img_paths):
        img_filename = os.path.basename(img_path)
        if index < num_val:
            # val
            shutil.copy(img_path, os.path.join(val_dir, img_filename))
        elif index < num_val + num_test:
            # test
            shutil.copy(img_path, os.path.join(test_dir, img_filename))
        else:
            # train
            shutil.copy(img_path, os.path.join(train_dir, img_filename))


def split_yolo_labels(input_dir, split_images_dir, output_dir):
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        return

    for split in [TRAIN_DIRNAME, VAL_DIRNAME, TEST_DIRNAME]:
        output_split_dir = os.path.join(output_dir, split)
        if not os.path.exists(output_split_dir):
            os.makedirs(output_split_dir)
        images_dir = os.path.join(split_images_dir, split)
        img_paths = glob(os.path.join(images_dir, "*.jpg")) + glob(
            os.path.join(images_dir, "*.png")
        )
        for img_path in img_paths:
            img_filename = os.path.basename(img_path)
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(input_dir, label_filename)
            shutil.copy(label_path, os.path.join(output_split_dir, label_filename))


def split_masks(input_dir, split_images_dir, output_dir):
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        return

    for split in [TRAIN_DIRNAME, VAL_DIRNAME, TEST_DIRNAME]:
        output_split_dir = os.path.join(output_dir, split)
        if not os.path.exists(output_split_dir):
            os.makedirs(output_split_dir)
        images_dir = os.path.join(split_images_dir, split)
        img_paths = glob(os.path.join(images_dir, "*.jpg")) + glob(
            os.path.join(images_dir, "*.png")
        )
        for img_path in img_paths:
            img_filename = os.path.basename(img_path)
            mask_dirname = os.path.splitext(img_filename)[0]
            mask_dir = os.path.join(input_dir, mask_dirname)
            shutil.copytree(mask_dir, os.path.join(output_split_dir, mask_dirname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset between training, validation, and test datasets"
    )

    subparsers = parser.add_subparsers(title="Available Commands", dest="command")

    images_parser = subparsers.add_parser(
        "images", help="Split images between training, validation, and test"
    )
    images_parser.add_argument(
        "--input_dir",
        default=os.path.join(DEFAULT_RAW_DATA_DIR, "images"),
    )
    images_parser.add_argument(
        "--output_dir", default=os.path.join(DEFAULT_PREPARED_DATA_DIR, "images")
    )
    images_parser.add_argument(
        "--val_ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
    )
    images_parser.add_argument(
        "--test_ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
    )

    yolo_labels_parser = subparsers.add_parser(
        "yolo_labels",
        help="Split YOLO label files between training, validation, and test to match images dir",
    )
    yolo_labels_parser.add_argument(
        "--input_dir",
        default=os.path.join(DEFAULT_RAW_DATA_DIR, "labels"),
    )
    yolo_labels_parser.add_argument(
        "--split_images_dir",
        default=os.path.join(DEFAULT_PREPARED_DATA_DIR, "images"),
    )
    yolo_labels_parser.add_argument(
        "--output_dir",
        default=os.path.join(DEFAULT_PREPARED_DATA_DIR, "labels"),
    )

    masks_parser = subparsers.add_parser(
        "masks",
        help="Split instanced masks directories between training, validation, and test to match images dir",
    )
    masks_parser.add_argument(
        "--input_dir",
        default=os.path.join(DEFAULT_RAW_DATA_DIR, "masks"),
    )
    masks_parser.add_argument(
        "--split_images_dir",
        default=os.path.join(DEFAULT_PREPARED_DATA_DIR, "images"),
    )
    masks_parser.add_argument(
        "--output_dir",
        default=os.path.join(DEFAULT_PREPARED_DATA_DIR, "masks"),
    )

    args = parser.parse_args()

    if args.command == "images":
        split_images(args.input_dir, args.output_dir, args.val_ratio, args.test_ratio)
    elif args.command == "yolo_labels":
        split_yolo_labels(args.input_dir, args.split_images_dir, args.output_dir)
    elif args.command == "masks":
        split_masks(args.input_dir, args.split_images_dir, args.output_dir)
    else:
        print("Invalid command.")
