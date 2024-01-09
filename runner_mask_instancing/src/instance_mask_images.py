""" File: instance_mask_images.py

Description: Split mask images into instances based on Labelbox polyline labels
"""

import argparse
import os
import labelbox as lb
import ndjson
import math
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
CLIENT = lb.Client(os.getenv("LABELBOX_API_KEY"))
PROJECT_NAME = "Runner Mask Instancing"

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data",
)
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUT_DIR = os.path.join(DATA_DIR, "instanced")


def instance_masks(
    labelbox_export_file,
    project_name=PROJECT_NAME,
    images_dir=IMAGES_DIR,
    output_dir=OUT_DIR,
):
    with open(labelbox_export_file, "r") as f:
        rows = ndjson.load(f)

    project = CLIENT.get_projects(where=lb.Project.name == project_name).get_one()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for row in rows:
        image_filename = row["data_row"]["global_key"]
        label = row["projects"][project.uid]["labels"][0]
        annotations = label["annotations"]["objects"]
        objects = [
            annotation
            for annotation in annotations
            if annotation["name"] == "runner_mask_instance_line"
        ]

        segments = []
        instances = []
        for object_idx, object in enumerate(objects):
            points = object["line"]
            for point_idx in range(len(points)):
                if point_idx == 0:
                    continue

                prev_point = points[point_idx - 1]
                curr_point = points[point_idx]
                segment = [
                    (prev_point["x"], prev_point["y"]),
                    (curr_point["x"], curr_point["y"]),
                ]
                segments.append(segment)
                instances.append(object_idx + 1)

        # For each pixel in the mask image assign an instance id
        image_path = os.path.join(images_dir, image_filename)
        image = Image.open(image_path)
        width, height = image.size
        pixels = list(image.getdata())
        pixel_instances = [0] * len(pixels)
        assigned_instances = set()
        for idx in range(len(pixels)):
            if pixels[idx] == 0:
                continue

            x = idx % width
            y = idx // width
            closest_segment_idx = closest_segment((x, y), segments)
            instance_id = instances[closest_segment_idx]
            pixel_instances[idx] = instance_id
            assigned_instances.add(instance_id)

        # Generate instanced mask images
        for instance_id in assigned_instances:
            mask = Image.new("L", (width, height), color=0)

            for idx in range(len(pixel_instances)):
                if pixel_instances[idx] == instance_id:
                    x = idx % width
                    y = idx // width
                    mask.putpixel((x, y), 255)

            # Save mask to file
            mask_dir = os.path.join(output_dir, os.path.splitext(image_filename)[0])
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            mask.save(os.path.join(mask_dir, f"{instance_id}.png"))


def distance_point_to_segment(point, segment):
    x, y = point
    x1, y1 = segment[0]
    x2, y2 = segment[1]

    dot_product = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)

    if dot_product <= 0:
        # The point is closest to the starting point of the segment
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2

    if dot_product >= squared_length:
        # The point is closest to the ending point of the segment
        return math.sqrt((x - x2) ** 2 + (y - y2) ** 2)

    # The point is closest to a point on the line segment
    projection_factor = dot_product / squared_length
    projected_x = x1 + projection_factor * (x2 - x1)
    projected_y = y1 + projection_factor * (y2 - y1)

    return math.sqrt((x - projected_x) ** 2 + (y - projected_y) ** 2)


def closest_segment(point, segments):
    if not segments:
        return None

    for segment_idx, segment in enumerate(segments):
        distance = distance_point_to_segment(point, segment)
        if segment_idx == 0 or distance < min_distance:
            min_distance = distance
            closest_segment_idx = segment_idx

    return closest_segment_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split mask images into instances based on Labelbox polyline labels"
    )
    parser.add_argument(
        "-f",
        "--labelbox_export_file",
        required=True,
        help="File to Labelbox export",
    )
    parser.add_argument(
        "-n",
        "--project_name",
        default=PROJECT_NAME,
        help="Labelbox project name",
    )
    parser.add_argument(
        "-i",
        "--images_dir",
        default=IMAGES_DIR,
        help="Path to the directory containing images",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=OUT_DIR,
        help="Path to where the instanced masks will be written to",
    )
    args = parser.parse_args()
    instance_masks(
        args.labelbox_export_file, args.project_name, args.images_dir, args.output_dir
    )
