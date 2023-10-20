""" File: labelbox_api.py

Description: Functions for using the labelbox API and creating yolo model segmentation labels. 
Using these functions will require either creating a CLI application or creating a script. 
"""

import labelbox as lb
import labelbox.types as lb_types
from glob import glob
import os
import uuid
import cv2
import numpy as np
import ndjson
import urllib.request
from PIL import Image

# Use API key from a .env file, currently pip install dotenv causes an error
API_key = ""
client = lb.Client(API_key)

# Add a directory path for where images are stored
img_dir = "../data_store/segmentation_data/raw/"
# Add directory path for where existing mask labels are
mask_dir = "../data_store/segmentation_data/mask/"
# Add directory path for where yolo labels are
label_dir = "../data_store/segmentation_data/yolo_labels/"

project_name = "Runner Segmentation"
project = client.get_projects(where=lb.Project.name == project_name).get_one()
class_map = {"Runner": 0}


def import_images(dataset_name):
    dataset = client.get_datasets(where=lb.Dataset.name == dataset_name).get_one()
    if not dataset:
        dataset = client.create_dataset(name=dataset_name)
    global_keys = []
    img_paths = glob(os.path.join(img_dir, "*.jpg"))
    for img_path in img_paths:
        _, img_name = os.path.split(img_path)
        dataset.create_data_row({"row_data": img_path, "global_key": img_name})
        global_keys.append(img_name)
    return global_keys


def create_batch(batch_name, global_keys):
    batch = project.create_batch(
        batch_name,  # each batch in a project must have a unique name
        global_keys=global_keys,  # paginated collection of data row objects, list of data row ids or global keys
        priority=5,  # priority between 1(highest) - 5(lowest)
    )


def create_segmentation_annotation_from_mask(
    segment_name, color=(255, 255, 255), mask_size=(960, 720)
):
    """Given segmenation mask imags, create labelbox segmentation annotations. These can then be used as benchmark labels."""
    mask_paths = glob(os.path.join(mask_dir, ".png"))
    labels = []
    for mask_path in mask_paths:
        _, mask_name = os.path.split(mask_path)
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, mask_size)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_data = lb_types.MaskData(arr=mask)
        mask_annotation = lb_types.ObjectAnnotation(
            name=segment_name, value=lb_types.Mask(mask=mask_data, color=color)
        )
        img_name = mask_name.split(".")[0] + ".jpg"
        labels.append(
            lb_types.Label(
                data=lb_types.ImageData(global_key=img_name),
                annotations=[mask_annotation],
            )
        )
    try:
        upload_job = lb.LabelImport.create_from_objects(
            client=client,
            project_id=project.uid,
            name="label_import_job" + str(uuid.uuid4()),
            labels=labels,
        )
        upload_job.wait_until_done()
        print(f"successfully uploaded image {img_name}")
    except Exception as exc:
        print(f"Failed on image {img_name} with exception {exc}")


def create_polygon_annotation_from_yolo_label(annotation_name):
    """Given yolo labels, create labelbox polygon annotations. These can then be used as benchmark labels."""
    label_paths = glob(os.path.join(label_dir, "*.txt"))
    for label_path in label_paths:
        labels = []
        _, label_name = os.path.split(label_path)
        img_name = label_name.split(".")[0] + ".jpg"
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        with open(label_path, "r") as f:
            annotations = []
            height, width, _ = img.shape
            lines = f.readlines()
            # Create an annotation for each label
            for line in lines:
                points = []
                res = line.strip().replace("  ", " ").split(" ")
                cat = res.pop(0)
                while res:
                    x = float(res.pop(0)) * width
                    y = float(res.pop(0)) * height
                    points.append(lb_types.Point(x=x, y=y))
                polygon_annotation = lb_types.ObjectAnnotation(
                    name=annotation_name, value=lb_types.Polygon(points=points)
                )
                annotations.append(polygon_annotation)
            labels.append(
                lb_types.Label(
                    data=lb_types.ImageData(global_key=img_name),
                    annotations=annotations,
                )
            )
        try:
            upload_job = lb.LabelImport.create_from_objects(
                client=client,
                project_id=project.uid,
                name="label_import_job" + str(uuid.uuid4()),
                labels=labels,
            )
            upload_job.wait_until_done()
            print(f"successfully uploaded image {img_name}")
        except Exception as exc:
            print(f"Failed on image {img_name} with exception {exc}")


def create_yolo_labels_from_segment_ndjson(filepath, class_map):
    """Given a mask label from labelbox export, create yolo segmentation model training files"""
    with open(filepath, "r") as f:
        labels = ndjson.load(f)
    # should be a label for each datarow
    for label in labels:
        # This should change to global_id just to be internally consistent
        base_image = label["data_row"]["external_id"]
        yolo_label_name = os.path.splitext(base_image)[0] + ".txt"
        yolo_label_path = os.path.join(label_dir, yolo_label_name)
        with open(yolo_label_path, "w") as yolo_file:
            label_type = label["projects"][project.uid]["labels"]
            # Multiple label types per image needs more support
            if len(label_type) > 1:
                import pdb

                pdb.set_trace()
            annotations = label_type[0]["annotations"]["objects"]
            for annotation in annotations:
                mask_url = annotation["mask"]["url"]
                classification = class_map[annotation["name"]]
                req = urllib.request.Request(mask_url, headers=client.headers)
                # Should be a way to go directly into a numpy array
                pil_image = Image.open(urllib.request.urlopen(req))
                image = np.array(pil_image)
                # Single channel image, so RGB-BGR is not needed
                contours = cv2.findContours(
                    image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                height, width = image.shape
                for contour in contours[0]:
                    yolo_file.write(f"{int(classification)}")
                    for point in contour:
                        yolo_file.write(f" {point[0][0]/width} {point[0][1]/height}")
                    yolo_file.write("\n")
