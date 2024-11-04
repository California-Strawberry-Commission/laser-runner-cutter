import os
import argparse
from ultralytics import settings, YOLO
from time import perf_counter
import json
import cv2
import numpy as np
from glob import glob
from natsort import natsorted

PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_INPUT_IMAGE_SIZE = (1024, 768)
DEFAULT_EPOCHS = 150


class Yolo:
    def __init__(self, weights_file=None, input_image_size=DEFAULT_INPUT_IMAGE_SIZE):
        if weights_file is not None and os.path.exists(weights_file):
            print(f"Using weights file {weights_file}")
            self.model = YOLO(weights_file, task="segment")
        else:
            print(
                f"Weights file not defined or could not be found. Using model yaml file"
            )
            self.model = YOLO("yolov8n-seg.yaml", task="segment")
        # Note: for 'train' and 'val' modes, imgsz must be an integer, while for 'predict',
        # 'track', and 'export' modes, imgsz must be an (h, w) tuple or an integer. For 'train'
        # and 'val' modes, the largest dimension will be used.
        self.imgsz = (input_image_size[1], input_image_size[0])

    def load_weights(self, weights_file):
        self.model = YOLO(weights_file, task="segment")

    def train(
        self,
        dataset_yml,
        epochs=DEFAULT_EPOCHS,
    ):
        train_start = perf_counter()
        self.model.train(
            data=dataset_yml,
            imgsz=self.imgsz,
            device=0,
            batch=-1,
            epochs=epochs,
            flipud=0.5,
        )
        train_stop = perf_counter()
        print(f"Training finished in {train_stop - train_start} seconds.")

    def eval(self, dataset_yml):
        metrics = self.model.val(
            data=dataset_yml,
            imgsz=self.imgsz,
            device=0,
            batch=1,
            split="test",
            iou=0.6,
        )
        return metrics

    def predict(self, image, iou=0.6):
        """
        Run inference on an image and return bounding boxes and masks of detected object instances.

        Args:
            image (np.ndarray): color image in RGB8 format
            iou (float): Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
        """
        # YOLO prediction takes an numpy array with BGR8 format
        result = self.model(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            imgsz=self.imgsz,
            iou=iou,
            half=True,
            verbose=False,
        )[0]
        out = {}
        out["conf"] = result.boxes.conf.cpu().numpy()
        out["bboxes"] = result.boxes.xyxy.cpu().numpy()
        if result.masks is not None:
            out["masks"] = result.masks.xy

        return out

    def track(self, image, iou=0.6):
        """
        Run inference on an image and return bounding boxes, masks, and IDs of detected object instances.
        This differs from `predict` in that it also runs object tracking that maintains a unique ID for each detected object.

        Args:
            image (np.ndarray): color image in RGB8 format
            iou (float): Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
        """
        # YOLO prediction takes an numpy array with BGR8 format
        result = self.model.track(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            imgsz=self.imgsz,
            tracker="bytetrack.yaml",
            iou=iou,
            persist=True,
            half=True,
            verbose=False,
        )[0]
        out = {}
        out["conf"] = result.boxes.conf.cpu().numpy()
        out["bboxes"] = result.boxes.xyxy.cpu().numpy()
        out["track_ids"] = (
            result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []
        )
        if result.masks is not None:
            out["masks"] = result.masks.xy

        return out

    def debug(self, image_path, iou=0.6, show_result=False):
        if os.path.isfile(image_path):
            image_paths = [image_path]
        else:
            image_paths = natsorted(
                glob(os.path.join(image_path, "*.jpg"))
                + glob(os.path.join(image_path, "*.png"))
            )

        for path in image_paths:
            image = cv2.imread(path)
            # Resize image prior to inference for faster inference
            image = cv2.resize(
                image, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_LINEAR
            )
            image_array = np.array(image)

            # Measure inference time
            # Warmup
            for i in range(5):
                self.model.track(
                    image_array,
                    imgsz=self.imgsz,
                    tracker="bytetrack.yaml",
                    iou=iou,
                    persist=True,
                    half=True,
                )
            num_inferences = 10
            total_time_secs = 0
            for i in range(num_inferences):
                inference_start = perf_counter()
                self.model.track(
                    image_array,
                    imgsz=self.imgsz,
                    tracker="bytetrack.yaml",
                    iou=iou,
                    persist=True,
                    half=True,
                )
                total_time_secs = total_time_secs + (perf_counter() - inference_start)
            print(f"Average inference time: {total_time_secs / num_inferences} seconds")

            if show_result:
                result = self.model(image_array, imgsz=self.imgsz, iou=iou, half=True)[
                    0
                ]

                conf = result.boxes.conf.cpu().numpy()
                boxes = result.boxes.xywh
                if result.masks:
                    masks = result.masks.xy

                result.show()  # display to screen


def tuple_type(arg_string):
    try:
        # Parse the input string as a tuple
        parsed_tuple = tuple(map(int, arg_string.strip("()").split(",")))
        return parsed_tuple
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {arg_string}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO training, evaluation, and inference"
    )

    subparsers = parser.add_subparsers(title="Available Commands", dest="command")

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--dataset_yml", default=os.path.join(PROJECT_PATH, "dataset.yml")
    )
    train_parser.add_argument("--weights_file")
    train_parser.add_argument(
        "--input_image_size",
        type=tuple_type,
        default=f"({DEFAULT_INPUT_IMAGE_SIZE[0]}, {DEFAULT_INPUT_IMAGE_SIZE[1]})",
        help="(width, height) tuple",
    )
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)

    eval_parser = subparsers.add_parser("eval", help="Evaluate model performance")
    eval_parser.add_argument(
        "--dataset_yml", default=os.path.join(PROJECT_PATH, "dataset.yml")
    )
    eval_parser.add_argument("--weights_file")
    eval_parser.add_argument(
        "--input_image_size",
        type=tuple_type,
        default=f"({DEFAULT_INPUT_IMAGE_SIZE[0]}, {DEFAULT_INPUT_IMAGE_SIZE[1]})",
        help="(width, height) tuple",
    )

    debug_parser = subparsers.add_parser("debug", help="Debug model predictions")
    debug_parser.add_argument("--weights_file")
    debug_parser.add_argument(
        "--input_image_size",
        type=tuple_type,
        default=f"({DEFAULT_INPUT_IMAGE_SIZE[0]}, {DEFAULT_INPUT_IMAGE_SIZE[1]})",
        help="(width, height) tuple",
    )
    debug_parser.add_argument(
        "--image_path", required=True, help="Image file or dir path"
    )
    debug_parser.add_argument(
        "--show_result", action="store_true", help="Show inference results"
    )

    args = parser.parse_args()

    settings.update(
        {
            "datasets_dir": PROJECT_PATH,
            "runs_dir": os.path.join(PROJECT_PATH, "output", "ultralytics", "runs"),
            "weights_dir": os.path.join(
                PROJECT_PATH, "output", "ultralytics", "weights"
            ),
        }
    )

    model = Yolo(args.weights_file, args.input_image_size)
    if args.command == "train":
        model.train(args.dataset_yml, args.epochs)
    elif args.command == "eval":
        metrics = model.eval(args.dataset_yml)
        summary = {
            "mAP50 (box)": metrics.box.map50,
            "mAP50-95 (box)": metrics.box.map,
            "mAP50 (seg)": metrics.seg.map50,
            "mAP50-95 (seg)": metrics.seg.map,
        }
        print(json.dumps(summary))
    elif args.command == "debug":
        model.debug(args.image_path, show_result=args.show_result)
    else:
        print("Invalid command.")
