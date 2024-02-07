""" File: labelbox_api.py

Description: Functions for using the Labelbox API and creating yolo model labels from
Labelbox exports. 
"""

import argparse
from glob import glob
import os
import labelbox as lb
import labelbox.types as lb_types
import ndjson
from PIL import Image
from dotenv import load_dotenv
import uuid
from natsort import natsorted


load_dotenv()
CLIENT = lb.Client(os.getenv("LABELBOX_API_KEY"))
DATASET_NAME = "laser_detection"
PROJECT_NAME = "Laser Detection"
CLASS_MAP = {"laser": 0}

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/raw",
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
    img_paths = natsorted(img_paths)
    for img_path in img_paths:
        _, img_name = os.path.split(img_path)
        dataset.create_data_row({"row_data": img_path, "global_key": img_name})


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
                class_id = CLASS_MAP[object["name"]]
                # YOLO format: (class_id, x_center, y_center, width, height)
                # Set arbitrarily small bounding box size
                bb_size = (10, 10)
                yolo_label_file.write(
                    f"{class_id} {point['x'] / width} {point['y'] / height} {bb_size[0] / width} {bb_size[1] / height}\n"
                )


def upload_labels_from_yolo_labels(
    label_dir=LABELS_DIR, img_dir=IMAGES_DIR, project_name=PROJECT_NAME
):
    """Given YOLO labels, create labelbox point annotations."""
    label_paths = glob(os.path.join(label_dir, "*.txt"))
    for label_path in label_paths:
        _, label_filename = os.path.split(label_path)
        img_filename = os.path.splitext(label_filename)[0] + ".jpg"
        img_path = os.path.join(img_dir, img_filename)
        width, height = _get_image_size(img_path)

        labels = []
        with open(label_path, "r") as f:
            lines = f.readlines()
            annotations = []
            for line in lines:
                tokens = line.strip().split()
                x = float(tokens[1]) * width
                y = float(tokens[2]) * height
                annotations.append(
                    lb_types.ObjectAnnotation(
                        name="laser",
                        value=lb_types.Point(x=x, y=y),
                    )
                )
            if len(annotations) > 0:
                labels.append(
                    lb_types.Label(
                        data=lb_types.ImageData(global_key=img_filename),
                        annotations=annotations,
                    )
                )

        try:
            project = CLIENT.get_projects(
                where=lb.Project.name == project_name
            ).get_one()
            upload_job = lb.LabelImport.create_from_objects(
                client=CLIENT,
                project_id=project.uid,
                name="label_import_job" + str(uuid.uuid4()),
                labels=labels,
            )
            upload_job.wait_until_done()
            print(f"Successfully uploaded labels for {img_filename}")
        except Exception as ex:
            print(f"Failed to upload labels for {img_filename}: {ex}")


def _get_image_size(img_path):
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            return width, height
    except IOError:
        print(f"Failed to open the image at {img_path}")
        return None, None


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

    upload_labels_parser = subparsers.add_parser(
        "upload_labels", help="Upload Labelbox annotations from YOLO labels"
    )
    upload_labels_parser.add_argument(
        "-l",
        "--labels_dir",
        default=LABELS_DIR,
        help="Path to the directory containing labels",
    )
    upload_labels_parser.add_argument(
        "-i",
        "--images_dir",
        default=IMAGES_DIR,
        help="Path to the directory containing images",
    )
    upload_labels_parser.add_argument(
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
    elif args.command == "upload_labels":
        upload_labels_from_yolo_labels(
            args.labels_dir,
            args.images_dir,
            args.project_name,
        )
    else:
        print("Invalid command.")
