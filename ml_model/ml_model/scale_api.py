"""File: scale_api.py

Description: Functions for using the scale API. Currently the laser runner removal project is planning on 
using primarily labelbox for external data labeling. Future use of the scale api will require a either creating 
a CLI utility, or creating an external script for calling this function. 
"""

import scaleapi
from scaleapi.tasks import TaskType
from glob import glob
import os
import cv2
import json
from dotenv import load_dotenv

load_dotenv()
client = scaleapi.ScaleClient(os.getenv("SCALE_API_KEY"))


def create_project(project_name):
    project = client.create_project(
        project_name=project_name,
        task_type=TaskType.ImageAnnotation,
        params={
            "annotation_attributes": {},
            "geometries": {"polygon": {"objects_to_annotate": ["Runner"]}},
        },
        rapid=True,
    )
    return project


def create_batch(project_name, batch_name):
    client.create_batch(
        project=project_name, batch_name=batch_name, self_label_batch=True
    )


# clear all task unique ids, only do this is you fubar
def fubar(project_name):
    tasks = client.get_tasks(project_name)
    for task in tasks:
        client.clear_task_unique_id(task.task_id)


def file_upload(project_name, img_path, img_name):
    # Upload the file
    with open(img_path, "rb") as img_f:
        uploaded_image = None
        try:
            uploaded_image = client.upload_file(
                img_f,
                project_name=project_name,
                display_name=img_name,
                reference_id=img_name,
                metadata=json.dumps({"filename": img_name}),
            )
            print(f"Uploaded Image{img_name}")
        except Exception as exc:
            print(exc)
    return uploaded_image


def create_tasks(project_name, batch_name, dir_name):
    files = glob(os.path.join(dir_name, "*"))
    for file in files:
        _, tail = os.path.split(file)
        uploaded_file = file_upload(project_name, file, tail)
        payload = {
            "project": project_name,
            "attachment": uploaded_file.attachment_url,
            "batch": batch_name,
            "metadata": {"filename": tail},
            "unique_id": tail,
        }
        try:
            ret = client.create_task(TaskType.SegmentAnnotation, **payload)
            print(f"uploaded image {tail} to scale.ai")
        except scaleapi.exceptions.ScaleDuplicateResource:
            print(f"File {tail} already uploaded to scale.ai")


def create_evaluation_task(project_name, yolo_label_dir, img_dir):
    """
    Create scale evaluation tasks. The evaluation tasks are used to calculate how well
    external labelers match the expected labels from internal team.
    """
    # ToDo: Current this only supports annotations of type polygon with a single
    # class of name runner
    yolo_labels = glob(os.path.join(yolo_label_dir, "*.txt"))
    for label in yolo_labels:
        expected_results = {"annotations": []}
        _, label_name = os.path.split(label)
        img_name = label_name.split(".")[0] + ".jpg"
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        with open(label, "r") as f:
            height, width, _ = img.shape
            lines = f.readlines()
            # Create an annotation for each label
            for line in lines:
                result = {}
                result["label"] = "Runner"
                verts = []
                res = line.strip().replace("  ", " ").split(" ")
                cat = res.pop(0)
                while res:
                    x = float(res.pop(0)) * width
                    y = float(res.pop(0)) * height
                    verts.append({"x": x, "y": y})
                result["vertices"] = verts
                result["type"] = "polygon"
                expected_results["annotations"].append(result)
            uploaded_file = file_upload(project_name, img_path, img_name)
            try:
                if uploaded_file is not None:
                    client.create_evaluation_task(
                        TaskType.ImageAnnotation,
                        attachment=uploaded_file.attachment_url,
                        project=project_name,
                        expected_response=expected_results,
                        unique_id=img_name,
                        metadata={
                            "filename": img_name,
                        },
                    )
                    print(f"Created evaluation task for {img_name}")
            except Exception as exc:
                print(exc)
