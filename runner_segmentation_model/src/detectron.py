import os
import argparse
from glob import glob
import math
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2
import numpy as np
import pycocotools
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_PREPARED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/prepared/runner1800",
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_PATH, "output", "detectron")
DEFAULT_EPOCHS = 150

setup_logger()
matplotlib.use("tkagg")


def get_record(img_path, mask_subdir):
    record = {}

    height, width = cv2.imread(img_path).shape[:2]
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    record["file_name"] = img_path
    record["image_id"] = img_id
    record["height"] = height
    record["width"] = width

    # Create annotations from masks
    annotations = []
    mask_paths = glob(os.path.join(mask_subdir, "*.jpg")) + glob(
        os.path.join(mask_subdir, "*.png")
    )
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # Don't include masks without any valid pixel values
        if np.max(mask) < 255:
            continue

        # Calculate the bounding box
        nonzero_indices = np.argwhere(mask > 0)
        min_x = np.min(nonzero_indices[:, 1])
        max_x = np.max(nonzero_indices[:, 1])
        min_y = np.min(nonzero_indices[:, 0])
        max_y = np.max(nonzero_indices[:, 0])
        bbox = [min_x, min_y, max_x, max_y]

        annotation = {
            "bbox": bbox,
            "bbox_mode": detectron2.structures.BoxMode.XYXY_ABS,
            "segmentation": pycocotools.mask.encode(
                np.asarray(mask > 0, dtype=np.uint8, order="F")
            ),
            "category_id": 0,
        }
        annotations.append(annotation)

    record["annotations"] = annotations
    return record


def get_dataset(img_dir, mask_dir):
    """Custom dataset for Detectron2

    Args:
        img_dir (string): path to images directory
        mask_dir (string): path to mask directory. Note that this directory must
                           contain subdirectories with names matching the image
                           files in img_dir without the extension
    """
    img_paths = glob(os.path.join(img_dir, "*.jpg")) + glob(
        os.path.join(img_dir, "*.png")
    )

    dataset_dicts = []
    for img_path in tqdm(img_paths, desc="Creating dataset"):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        mask_subdir = os.path.join(mask_dir, img_id)
        dataset_dicts.append(get_record(img_path, mask_subdir))

    return dataset_dicts


class Detectron:
    def __init__(self, output_dir=DEFAULT_OUTPUT_DIR):
        # Load default config
        # Config reference: https://detectron2.readthedocs.io/en/latest/modules/config.html#yaml-config-references
        cfg = get_cfg()
        cfg.OUTPUT_DIR = output_dir
        # See available models at https://github.com/facebookresearch/detectron2/tree/main/configs/COCO-InstanceSegmentation
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        # Initialize weights from model zoo by default
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        # Only has one class (runner). Note that this config param does not include background as a class.
        # See https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.DATALOADER.NUM_WORKERS = 2
        # Batch size
        cfg.SOLVER.IMS_PER_BATCH = 2
        # Learning rate
        cfg.SOLVER.BASE_LR = 0.0001
        # LR decay
        cfg.SOLVER.GAMMA = 0.1
        cfg.SOLVER.STEPS = np.arange(5000, 1000001, 5000).tolist()
        self.cfg = cfg

    def load_weights(self, weights_file):
        self.cfg.MODEL.WEIGHTS = weights_file

    def train(self, images_dir, masks_dir, epochs=DEFAULT_EPOCHS, resume=False):
        # Register datasets
        for d in ["train", "val"]:
            DatasetCatalog.register(
                f"runner_{d}",
                lambda images_dir=images_dir, masks_dir=masks_dir, d=d: get_dataset(
                    os.path.join(images_dir, d), os.path.join(masks_dir, d)
                ),
            )
            MetadataCatalog.get(f"runner_{d}").set(thing_classes=["runner"])

        self.cfg.DATASETS.TRAIN = ("runner_train",)
        # MAX_ITER = num_images / batch_size * epochs
        num_train_images = len(
            glob(os.path.join(images_dir, "train", "*.jpg"))
            + glob(os.path.join(images_dir, "train", "*.png"))
        )
        self.cfg.SOLVER.MAX_ITER = (
            math.ceil(num_train_images / self.cfg.SOLVER.IMS_PER_BATCH) * epochs
        )

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=resume)
        trainer.train()

    def eval(self, images_dir, masks_dir):
        # Register dataset
        DatasetCatalog.register(
            f"runner_test",
            lambda images_dir=images_dir, masks_dir=masks_dir: get_dataset(
                os.path.join(images_dir, "test"), os.path.join(masks_dir, "test")
            ),
        )
        MetadataCatalog.get(f"runner_test").set(thing_classes=["runner"])

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        predictor = DefaultPredictor(self.cfg)
        evaluator = COCOEvaluator("runner_test", output_dir=self.cfg.OUTPUT_DIR)
        data_loader = build_detection_test_loader(self.cfg, "runner_test")
        print(inference_on_dataset(predictor.model, data_loader, evaluator))

    def debug(self, image_file, mask_subdir=None):
        img = cv2.imread(image_file)
        # Custom confidence score threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor = DefaultPredictor(self.cfg)
        outputs = predictor(
            img
        )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        def show(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]
            fix, axes = plt.subplots(ncols=len(imgs), squeeze=False)
            for i, img in enumerate(imgs):
                axes[0, i].imshow(np.asarray(img))
                axes[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.show()

        images_to_display = []
        MetadataCatalog.get("runner_debug").set(thing_classes=["runner"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get("runner_debug"),
            scale=0.5,
        )
        if mask_subdir is not None:
            record = get_record(image_file, mask_subdir)
            gt_img = visualizer.draw_dataset_dict(record)
            gt_img = gt_img.get_image()[:, :, ::-1]
            images_to_display.append(gt_img)

        pred_img = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        pred_img = pred_img.get_image()[:, :, ::-1]
        images_to_display.append(pred_img)

        show(images_to_display)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detectron2 training, evaluation, and inference"
    )

    subparsers = parser.add_subparsers(title="Available Commands", dest="command")

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--images_dir",
        default=os.path.join(DEFAULT_PREPARED_DATA_DIR, "images"),
    )
    train_parser.add_argument(
        "--masks_dir",
        default=os.path.join(DEFAULT_PREPARED_DATA_DIR, "masks"),
    )
    train_parser.add_argument(
        "--weights_file",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "model_final.pth"),
    )
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train_parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate model performance")
    eval_parser.add_argument(
        "--images_dir",
        default=os.path.join(DEFAULT_PREPARED_DATA_DIR, "images"),
    )
    eval_parser.add_argument(
        "--masks_dir",
        default=os.path.join(DEFAULT_PREPARED_DATA_DIR, "masks"),
    )
    eval_parser.add_argument(
        "--weights_file",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "model_final.pth"),
    )
    eval_parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    debug_parser = subparsers.add_parser("debug", help="Debug model predictions")
    debug_parser.add_argument(
        "--weights_file",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "model_final.pth"),
    )
    debug_parser.add_argument("--image_file", required=True)
    debug_parser.add_argument("--mask_subdir")

    args = parser.parse_args()

    output_dir = None if args.command == "debug" else args.output_dir
    model = Detectron(output_dir=output_dir)
    weights_file = args.weights_file
    weights_file_exists = weights_file is not None and os.path.exists(weights_file)
    if weights_file_exists:
        model.load_weights(weights_file)

    if args.command == "train":
        model.train(
            args.images_dir,
            args.masks_dir,
            epochs=args.epochs,
            resume=weights_file_exists,
        )
    elif args.command == "eval":
        model.eval(args.images_dir, args.masks_dir)
    elif args.command == "debug":
        model.debug(args.image_file, mask_subdir=args.mask_subdir)
    else:
        print("Invalid command.")
