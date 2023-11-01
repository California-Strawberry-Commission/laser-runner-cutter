""" File: labelbox_api.py

Description: Functions for using the Labelbox API and creating yolo model labels from
Labelbox exports. 
"""

from glob import glob
import os
import labelbox as lb
import ndjson
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
client = lb.Client(os.getenv("LABELBOX_API_KEY"))
dataset_name = "laser_detection"
project_name = "Laser Detection"
project = client.get_projects(where=lb.Project.name == project_name).get_one()
class_map = {"laser": 0}

data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data_store/laser_detection",
)
img_dir = os.path.join(data_dir, "raw")
label_dir = os.path.join(data_dir, "raw_labels")


def import_images(dataset_name, img_dir=img_dir):
    dataset = client.get_datasets(where=lb.Dataset.name == dataset_name).get_one()
    if not dataset:
        dataset = client.create_dataset(name=dataset_name)
    global_keys = []
    img_paths = glob(os.path.join(img_dir, "*.jpg"))
    for img_path in img_paths:
        _, img_name = os.path.split(img_path)
        dataset.create_data_row({"row_data": img_path, "global_key": img_name})
        global_keys.append(img_name)


def create_yolo_labels_from_export_ndjson(export_filepath, label_outdir=label_dir):
    """Given a labelbox export, create yolo model label files"""
    with open(export_filepath, "r") as f:
        rows = ndjson.load(f)

    for row in rows:
        image_filename = row["data_row"]["global_key"]
        label = row["projects"][project.uid]["labels"][0]
        annotations = label["annotations"]["objects"]
        objects = [
            annotation for annotation in annotations if annotation["name"] == "laser"
        ]

        media_attributes = row["media_attributes"]
        height = media_attributes["height"]
        width = media_attributes["width"]

        yolo_label_name = os.path.splitext(image_filename)[0] + ".txt"
        yolo_label_path = os.path.join(label_outdir, yolo_label_name)

        with open(yolo_label_path, "w") as yolo_label_file:
            for object in objects:
                point = object["point"]
                class_id = class_map[object["name"]]
                # YOLO format: (class_id, x_center, y_center, width, height)
                # Set arbitrarily small bounding box size
                bb_size = (10, 10)
                yolo_label_file.write(
                    f"{class_id} {point['x'] / width} {point['y'] / height} {bb_size[0] / width} {bb_size[1] / height}\n"
                )
