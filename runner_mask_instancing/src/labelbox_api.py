""" File: labelbox_api.py

Description: Functions for using the Labelbox API and creating yolo model labels from
Labelbox exports. 
"""

import argparse
from glob import glob
import os
import labelbox as lb
import ndjson
from dotenv import load_dotenv


load_dotenv()
CLIENT = lb.Client(os.getenv("LABELBOX_API_KEY"))
DATASET_NAME = "Runner Masks"
PROJECT_NAME = "Runner Mask Instancing"
CLASS_MAP = {"runner_mask_instance_line": 0}

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data",
)
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")


def import_images(dataset_name=DATASET_NAME, images_dir=IMAGES_DIR):
    dataset = CLIENT.get_datasets(where=lb.Dataset.name == dataset_name).get_one()
    if not dataset:
        dataset = CLIENT.create_dataset(name=dataset_name)
    img_paths = glob(os.path.join(images_dir, "*.jpg")) + glob(
        os.path.join(images_dir, "*.png")
    )
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

    args = parser.parse_args()

    if args.command == "import_images":
        import_images(args.dataset_name, args.images_dir)
    elif args.command == "create_labels":
        create_yolo_labels_from_export_ndjson(
            args.labelbox_export_file, args.project_name, args.output_dir
        )
    else:
        print("Invalid command.")
