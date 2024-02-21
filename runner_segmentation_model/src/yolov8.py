import os
import argparse
from ultralytics import settings, YOLO
from time import perf_counter

PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_SIZE = (1024, 768)
DEFAULT_EPOCHS = 150


class YoloV8:
    def __init__(self, weights_file=None):
        if weights_file is not None and os.path.exists(weights_file):
            self.model = YOLO(weights_file)
        else:
            self.model = YOLO("yolov8n-seg.yaml")

    def load_weights(self, weights_file):
        self.model = YOLO(weights_file)

    def train(self, dataset_yml, size=DEFAULT_SIZE, epochs=DEFAULT_EPOCHS):
        train_start = perf_counter()
        self.model.train(
            data=dataset_yml,
            imgsz=size,
            device=0,
            batch=-1,
            epochs=epochs,
            flipud=0.5,
        )
        train_stop = perf_counter()
        print(f"Training finished in {train_stop - train_start} seconds.")

    def eval(self, dataset_yml):
        metrics = model.val(
            data=dataset_yml,
            split="test",
        )
        return metrics


def tuple_type(arg_string):
    try:
        # Parse the input string as a tuple
        parsed_tuple = tuple(map(int, arg_string.strip("()").split(",")))
        return parsed_tuple
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {arg_string}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask R-CNN training, evaluation, and inference"
    )

    subparsers = parser.add_subparsers(title="Available Commands", dest="command")

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--dataset_yml", default=os.path.join(PROJECT_PATH, "dataset.yml")
    )
    train_parser.add_argument(
        "--size", type=tuple_type, default=f"({DEFAULT_SIZE[0]}, {DEFAULT_SIZE[1]})"
    )
    train_parser.add_argument(
        "--weights_file", default=os.path.join(PROJECT_PATH, "models", "yolov8.pt")
    )
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)

    eval_parser = subparsers.add_parser("eval", help="Evaluate model performance")
    eval_parser.add_argument(
        "--dataset_yml", default=os.path.join(PROJECT_PATH, "dataset.yml")
    )
    eval_parser.add_argument(
        "--weights_file", default=os.path.join(PROJECT_PATH, "models", "yolov8.pt")
    )

    args = parser.parse_args()

    settings.update(
        {
            "datasets_dir": PROJECT_PATH,
            "runs_dir": os.path.join(PROJECT_PATH, "ultralytics", "runs"),
            "weights_dir": os.path.join(PROJECT_PATH, "ultralytics", "weights"),
        }
    )

    weights_file = args.weights_file
    model = YoloV8(weights_file)
    if args.command == "train":
        model.train(args.dataset_yml, args.size, args.epochs)
    elif args.command == "eval":
        metrics = model.eval(args.dataset_yml)
        print(f"mAP50: {metrics.seg.map50}")
        print(f"mAP50-95: {metrics.seg.map}")
    else:
        print("Invalid command.")
