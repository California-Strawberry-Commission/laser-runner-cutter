import numpy as np
import glob
import os
import cv2

from shapely.geometry import Point
from PIL import Image


def split_mask(mask_dir, mask_subdir):
    mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path)
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        mask = mask[:, :, 0]
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        mask_name = os.path.splitext(os.path.split(mask_path)[-1])[0]
        new_subdir = os.path.join(mask_subdir, mask_name)

        os.mkdir(new_subdir)
        for idx in range(len(contours)):
            new_mask = np.zeros(mask.shape)
            cv2.drawContours(
                new_mask, contours, idx, color=(255, 255, 255), thickness=-1
            )
            new_path = os.path.join(new_subdir, f"{idx}.jpg")
            cv2.imwrite(new_path, new_mask)


def convert_images_to_png(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            # Check if the file is a .jpg image
            if filename.lower().endswith(".jpg"):
                try:
                    # Open the image using Pillow
                    with Image.open(filepath) as img:
                        # Convert and save as .png
                        new_filepath = os.path.splitext(filepath)[0] + ".png"
                        img.save(new_filepath, "PNG")

                    # Remove the original .jpg file
                    os.remove(filepath)
                    print(f"Converted {filename} to PNG and removed original.")
                except Exception as e:
                    print(f"Failed to convert {filename}: {e}")
            else:
                print(f"Skipping {filename} - Not a JPG image.")


def resize_images(folder_path, output_width, output_height):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            # Check if the file is an image
            if any(filename.lower().endswith(ext) for ext in [".jpg", ".png", ".jpeg"]):
                try:
                    # Open the image using Pillow
                    with Image.open(filepath) as img:
                        # Resize the image
                        resized_img = img.resize((output_width, output_height))

                        # Save the resized image with the same format
                        resized_img.save(filepath)

                    print(f"Resized {filename}.")
                except Exception as e:
                    print(f"Failed to resize {filename}: {e}")
            else:
                print(f"Skipping {filename} - Not an image.")


def binary_mask_to_points(binary_mask):
    points = []
    rows, cols = len(binary_mask), len(binary_mask[0])
    for i in range(rows):
        for j in range(cols):
            if binary_mask[i][j]:
                points.append(Point(j, i))  # Points are created with (x, y) coordinates
    return points


def find_closest_point(binary_mask, given_point):
    mask_points = binary_mask_to_points(binary_mask)
    given_point = Point(
        given_point[1], given_point[0]
    )  # Converting (x, y) to Point(y, x)
    closest_point = min(mask_points, key=lambda p: given_point.distance(p))
    return closest_point.x, closest_point.y  # Reverting