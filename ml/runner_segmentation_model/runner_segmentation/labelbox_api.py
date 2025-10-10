import argparse
import os
import uuid
from glob import glob

import cv2
import labelbox as lb
import labelbox.types as lb_types
import ndjson
import numpy as np
import requests
from dotenv import load_dotenv
from natsort import natsorted
from tqdm import tqdm

from ml_utils.segment_utils import convert_contour_to_mask

from .yolo import Yolo

load_dotenv()
CLIENT = lb.Client(os.getenv("LABELBOX_API_KEY"))
DEFAULT_DATASET_NAME = "Runner2024"
DEFAULT_PROJECT_NAME = "Runner2024 Segmentation"
CLASS_MAP = {"Runner": 0}

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data",
)
DEFAULT_IMAGES_DIR = os.path.join(DATA_DIR, "raw", "runner1800", "images")
DEFAULT_LABELS_DIR = os.path.join(DATA_DIR, "raw", "runner1800", "labels")
DEFAULT_MASKS_DIR = os.path.join(DATA_DIR, "raw", "runner1800", "masks")
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../models",
)
DEFAULT_MODEL_PATH = os.path.join(
    MODELS_DIR, "runner1800-yolov8m-seg", "weights", "best.pt"
)
DEFAULT_MODEL_IMAGE_SIZE = (1024, 768)


def import_images(dataset_name=DEFAULT_DATASET_NAME, images_dir=DEFAULT_IMAGES_DIR):
    dataset = CLIENT.get_datasets(where=lb.Dataset.name == dataset_name).get_one()
    if not dataset:
        dataset = CLIENT.create_dataset(name=dataset_name)

    img_paths = glob(os.path.join(images_dir, "*.jpg")) + glob(
        os.path.join(images_dir, "*.png")
    )
    img_paths = natsorted(img_paths)
    data_rows = []
    for img_path in img_paths:
        _, img_name = os.path.split(img_path)
        data_rows.append(
            {"row_data": img_path, "global_key": img_name, "external_id": img_name}
        )
    print(f"Uploading {len(data_rows)} images. This may take a while...")
    upload_task = dataset.create_data_rows(data_rows)
    upload_task.wait_till_done()
    print(f"Errors: {upload_task.errors}")
    print(f"Result: {len(upload_task.result)} images successfully uploaded")


def upload_yolo_predictions(
    images_dir=DEFAULT_IMAGES_DIR,
    model_path=DEFAULT_MODEL_PATH,
    model_image_size=DEFAULT_MODEL_IMAGE_SIZE,
    project_name=DEFAULT_PROJECT_NAME,
    conf_threshold=0.0,
):
    project = CLIENT.get_projects(where=lb.Project.name == project_name).get_one()
    model = Yolo(model_path)
    predictions = []
    img_paths = glob(os.path.join(images_dir, "*.jpg")) + glob(
        os.path.join(images_dir, "*.png")
    )
    img_paths = natsorted(img_paths)
    for img_path in tqdm(img_paths):
        _, img_name = os.path.split(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_width = img.shape[1]
        img_height = img.shape[0]
        # Resize image prior to prediction to improve accuracy
        img = cv2.resize(img, model_image_size, interpolation=cv2.INTER_LINEAR)
        result_width = model_image_size[0]
        result_height = model_image_size[1]

        result = model.predict(img)

        result_conf = result["conf"]
        for idx in range(result_conf.size):
            conf = result_conf[idx]
            if conf < conf_threshold:
                continue

            mask_contour = result["masks"][idx]
            # Scale the result coords to frame coords
            mask_contour[:, 0] *= img_width / result_width
            mask_contour[:, 1] *= img_height / result_height

            # Remove small contours from mask
            area_threshold = 64
            if cv2.contourArea(mask_contour) < area_threshold:
                continue

            mask = convert_contour_to_mask(
                mask_contour, mask_size=(img_width, img_height)
            )
            if mask is None:
                continue

            predictions.append(
                lb_types.Label(
                    data={"global_key": img_name},
                    annotations=[
                        lb_types.ObjectAnnotation(
                            name=list(CLASS_MAP.keys())[0],
                            confidence=conf,
                            value=lb_types.Mask(
                                mask=lb.types.MaskData.from_2D_arr(mask),
                                color=(255, 255, 255),
                            ),
                        )
                    ],
                )
            )

        print(f"Prediction generated for {img_name}")

    try:
        print(f"Uploading predictions...")
        upload_job = lb.MALPredictionImport.create_from_objects(
            client=CLIENT,
            project_id=project.uid,
            name="mal_job" + str(uuid.uuid4()),
            predictions=predictions,
        )
        upload_job.wait_till_done()
        print(f"Successfully uploaded predictions")
    except Exception as exc:
        print(f"Failed to upload predictions: {exc}")


def create_masks_from_labelbox_export(
    labelbox_export_file, masks_dir=DEFAULT_MASKS_DIR, project_name=DEFAULT_PROJECT_NAME
):
    with open(labelbox_export_file, "r") as f:
        rows = ndjson.load(f)

    project = CLIENT.get_projects(where=lb.Project.name == project_name).get_one()
    api_key = os.getenv("LABELBOX_API_KEY")

    for row in tqdm(rows):
        data_row = row["data_row"]
        # Prefer external_id, fallback to global_key for the image name
        image_filename = data_row.get("external_id") or data_row["global_key"]
        mask_subdir = os.path.join(masks_dir, os.path.splitext(image_filename)[0])
        if not os.path.exists(mask_subdir):
            os.makedirs(mask_subdir)

        label = row["projects"][project.uid]["labels"][0]
        objects = [
            annotation
            for annotation in label["annotations"]["objects"]
            if annotation["name"] == "Runner"
        ]
        for idx, object in enumerate(objects):
            url = object["mask"]["url"]
            # Download mask
            headers = {"Authorization": api_key}
            with requests.get(url, headers=headers, stream=True) as r:
                try:
                    r.raw.decode_content = True
                    mask = np.asarray(bytearray(r.raw.read()), dtype="uint8")
                    mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    output_file = os.path.join(mask_subdir, f"{idx}.png")
                    cv2.imwrite(output_file, mask)
                except Exception as exc:
                    print(f"Failed to save mask for {image_filename}: {exc}")


def download_images(labelbox_export_file, output_dir):
    with open(labelbox_export_file, "r") as f:
        rows = ndjson.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for row in tqdm(rows):
        try:
            data_row = row["data_row"]
            # Prefer external_id, fallback to global_key for the image name
            image_filename = data_row.get("external_id") or data_row["global_key"]
            image_url = data_row.get("row_data")
            if image_url is None:
                continue
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(resp.content)
            print(f"Downloaded: {image_url}")
        except Exception as e:
            print(f"Error attempting to download image: {e}")


def tuple_type(arg_string):
    try:
        # Parse the input string as a tuple
        parsed_tuple = tuple(map(int, arg_string.strip("()").split(",")))
        return parsed_tuple
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {arg_string}")


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
        default=DEFAULT_DATASET_NAME,
        help="Labelbox dataset name",
    )
    import_parser.add_argument(
        "-i",
        "--images_dir",
        default=DEFAULT_IMAGES_DIR,
        help="Path to the directory containing images",
    )

    upload_predictions_parser = subparsers.add_parser(
        "upload_predictions", help="Upload YOLO predictions to Labelbox"
    )
    upload_predictions_parser.add_argument(
        "-i",
        "--input_dir",
        default=DEFAULT_IMAGES_DIR,
        help="Path to the directory containing images",
    )
    upload_predictions_parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to trained runner mask instancing YOLOv8 model",
    )
    upload_predictions_parser.add_argument(
        "-s",
        "--model_image_size",
        type=tuple_type,
        default=f"({DEFAULT_MODEL_IMAGE_SIZE[0]}, {DEFAULT_MODEL_IMAGE_SIZE[1]})",
        help="Image size that the YOLOv8 model was trained with",
    )
    upload_predictions_parser.add_argument(
        "-n",
        "--project_name",
        default=DEFAULT_PROJECT_NAME,
        help="Labelbox project name",
    )

    create_masks_parser = subparsers.add_parser(
        "create_masks",
        help="Create binary mask image files from Labelbox annotation export file",
    )
    create_masks_parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help="Path to the Labelbox export ndjson file",
    )
    create_masks_parser.add_argument(
        "-o",
        "--output_dir",
        default=DEFAULT_MASKS_DIR,
        help="Path to write the binary mask image files",
    )
    create_masks_parser.add_argument(
        "-n",
        "--project_name",
        default=DEFAULT_PROJECT_NAME,
        help="Labelbox project name",
    )

    download_parser = subparsers.add_parser(
        "download_images",
        help="Download images from Labelbox",
    )
    download_parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help="Path to the Labelbox export ndjson file",
    )
    download_parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to the directory to save images",
    )

    args = parser.parse_args()

    if args.command == "import_images":
        import_images(args.dataset_name, args.images_dir)
    elif args.command == "upload_predictions":
        upload_yolo_predictions(
            args.input_dir, args.model, args.model_image_size, args.project_name
        )
    elif args.command == "create_masks":
        create_masks_from_labelbox_export(
            args.input_file, args.output_dir, args.project_name
        )
    elif args.command == "download_images":
        download_images(args.input_file, args.output_dir)
    else:
        print("Invalid command.")
