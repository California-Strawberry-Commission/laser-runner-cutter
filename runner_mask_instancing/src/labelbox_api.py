""" File: labelbox_api.py

Description: Functions for using the Labelbox API and creating yolo model labels from
Labelbox exports. 
"""

import argparse
from glob import glob
import os
import uuid
import numpy as np
import labelbox as lb
import labelbox.types as lb_types
import ndjson
from dotenv import load_dotenv
from natsort import natsorted
from segment_utils import convert_contour_to_line_segments
from ultralytics import YOLO
from PIL import Image


load_dotenv()
CLIENT = lb.Client(os.getenv("LABELBOX_API_KEY"))
DATASET_NAME = "Runner1800 Masks"
PROJECT_NAME = "Runner1800 Mask Instancing"
CLASS_MAP = {"runner_mask_instance_line": 0}

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data",
)
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../models",
)
MODEL_PATH = os.path.join(MODELS_DIR, "yolov8-segment", "weights", "best.pt")


def import_images(dataset_name=DATASET_NAME, images_dir=IMAGES_DIR):
    dataset = CLIENT.get_datasets(where=lb.Dataset.name == dataset_name).get_one()
    if not dataset:
        dataset = CLIENT.create_dataset(name=dataset_name)

    img_paths = glob(os.path.join(images_dir, "*.jpg")) + glob(
        os.path.join(images_dir, "*.png")
    )
    img_paths = natsorted(img_paths)
    for img_path in img_paths:
        _, img_name = os.path.split(img_path)
        dataset.create_data_row({"row_data": img_path, "global_key": img_name})
        print(f"Uploaded {img_name}")


def create_yolo_labels_from_export_ndjson(
    export_filepath, project_name=PROJECT_NAME, label_outdir=LABELS_DIR
):
    """Given a labelbox export, create yolo model label files"""
    with open(export_filepath, "r") as f:
        rows = ndjson.load(f)

    project = CLIENT.get_projects(where=lb.Project.name == project_name).get_one()

    for row in rows:
        image_filename = row["data_row"]["global_key"]
        label = row["projects"][project.uid]["labels"][0]
        annotations = label["annotations"]["objects"]
        objects = [
            annotation
            for annotation in annotations
            if annotation["name"] == "runner_mask_instance_line"
        ]

        media_attributes = row["media_attributes"]
        height = media_attributes["height"]
        width = media_attributes["width"]

        yolo_label_name = os.path.splitext(image_filename)[0] + ".txt"
        yolo_label_path = os.path.join(label_outdir, yolo_label_name)

        with open(yolo_label_path, "w") as yolo_label_file:
            for object in objects:
                # YOLO format: (class_id, x1, y1, x2, y2, ..., xn, yn)
                # x and y are normalized on image dimensions
                points = object["line"]
                class_id = CLASS_MAP[object["name"]]
                yolo_label_file.write(f"{class_id}")
                for point in points:
                    yolo_label_file.write(
                        f" {point['x'] / width} {point['y'] / height}"
                    )
                yolo_label_file.write("\n")


def upload_yolo_predictions(
    masks_dir=IMAGES_DIR, model_path=MODEL_PATH, project_name=PROJECT_NAME
):
    project = CLIENT.get_projects(where=lb.Project.name == project_name).get_one()
    model = YOLO(model_path)
    mask_paths = glob(os.path.join(masks_dir, "*.jpg")) + glob(
        os.path.join(masks_dir, "*.png")
    )
    mask_paths = natsorted(mask_paths)
    for mask_path in mask_paths:
        img_name = os.path.split(mask_path)[1]
        image = Image.open(mask_path)
        width, height = image.size
        results = model.predict(
            mask_path,
            imgsz=(width, height),
            iou=0.5,
            conf=0.3,
        )

        predictions = []
        for result in results:
            for c in result:
                contour = c.masks.xy.pop()
                contour = contour.astype(np.int32)
                points = convert_contour_to_line_segments(
                    contour, result.orig_img.shape[:2]
                )
                if len(points) < 2:
                    continue

                predictions.append(
                    lb_types.Label(
                        data=lb_types.ImageData(global_key=img_name),
                        annotations=[
                            lb_types.ObjectAnnotation(
                                name=list(CLASS_MAP.keys())[0],
                                confidence=c.boxes.conf.cpu().numpy()[0],
                                value=lb_types.Line(
                                    points=[
                                        lb_types.Point(x=point[0], y=point[1])
                                        for point in points
                                    ]
                                ),
                            )
                        ],
                    )
                )

        try:
            upload_job = lb.MALPredictionImport.create_from_objects(
                client=CLIENT,
                project_id=project.uid,
                name="mal_job" + str(uuid.uuid4()),
                predictions=predictions,
            )
            upload_job.wait_until_done()
            print(f"successfully uploaded predictions for {img_name}")
        except Exception as exc:
            print(f"Failed to upload predictions for {img_name}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Functions for using the Labelbox API and creating yolo model labels from Labelbox exports.",
    )

    subparsers = parser.add_subparsers(title="Available Commands", dest="command")

    import_parser = subparsers.add_parser(
        "import_images",
        help="Import images to Labelbox from a local directory",
    )
    import_parser.add_argument(
        "-n",
        "--dataset_name",
        default=DATASET_NAME,
        help="Labelbox dataset name",
    )
    import_parser.add_argument(
        "-i",
        "--images_dir",
        default=IMAGES_DIR,
        help="Path to the directory containing images",
    )

    create_labels_parser = subparsers.add_parser(
        "create_labels", help="Create YOLO label files from a Labelbox ndjson export"
    )
    create_labels_parser.add_argument(
        "-f",
        "--labelbox_export_file",
        required=True,
        help="File to Labelbox export",
    )
    create_labels_parser.add_argument(
        "-n",
        "--project_name",
        default=PROJECT_NAME,
        help="Labelbox project name",
    )
    create_labels_parser.add_argument(
        "-o",
        "--output_dir",
        default=LABELS_DIR,
        help="Path to where the labels will be written to",
    )

    upload_predictions_parser = subparsers.add_parser(
        "upload_predictions", help="Upload YOLO predictions to Labelbox"
    )
    upload_predictions_parser.add_argument(
        "-i",
        "--input_dir",
        default=IMAGES_DIR,
        help="Path to the directory containing the instanced masks",
    )
    upload_predictions_parser.add_argument(
        "-m",
        "--model",
        default=MODEL_PATH,
        help="Path to trained runner mask instancing YOLOv8 model",
    )
    upload_predictions_parser.add_argument(
        "-n",
        "--project_name",
        default=PROJECT_NAME,
        help="Labelbox project name",
    )

    args = parser.parse_args()

    if args.command == "import_images":
        import_images(args.dataset_name, args.images_dir)
    elif args.command == "create_labels":
        create_yolo_labels_from_export_ndjson(
            args.labelbox_export_file, args.project_name, args.output_dir
        )
    elif args.command == "upload_predictions":
        upload_yolo_predictions(args.input_dir, args.model, args.project_name)
    else:
        print("Invalid command.")
