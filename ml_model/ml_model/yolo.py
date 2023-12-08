from abc import abstractmethod
import os
import tempfile
import yaml
from ml_model.model_base import ModelBase
from ultralytics import YOLO, settings
import glob
import random
import shutil


class YoloBaseModel(ModelBase):
    def __init__(self, weights=None, yaml=None):
        if weights is not None:
            self.model = YOLO(weights)
        elif yaml is not None:
            self.model = YOLO(yaml)
        else:
            self.model = YOLO(self.default_yaml())

        self.project_path = None
        # Temporary directories that contain symlinks to the base data to support the ultralytics input format
        self.temp_img_path = None
        # The label directory must be named label due to ultralytics requirements
        self.temp_label_path = None
        self.temp_data_yaml = None

    @abstractmethod
    def default_class_map():
        pass

    @abstractmethod
    def default_yaml():
        pass

    def create_temp_split(
        self, project_path, img_dir, label_dir, yolo_temp_dir=None, split_percentage=0.8
    ):
        # self.project_path = os.path.join(
        #    os.path.dirname(os.path.abspath(__file__)), data_folder
        # )
        self.project_path = project_path
        if yolo_temp_dir is None:
            self.yolo_temp_dir = os.path.join(self.project_path, self.name())
        else:
            self.yolo_temp_dir = yolo_temp_dir

        # Delete the temporary directory if it already exists.
        if os.path.exists(self.yolo_temp_dir):
            shutil.rmtree(self.yolo_temp_dir)

        train_imgs_dir = os.path.join(self.yolo_temp_dir, "images", "train")
        val_imgs_dir = os.path.join(self.yolo_temp_dir, "images", "val")
        train_labels_dir = os.path.join(self.yolo_temp_dir, "labels", "train")
        val_labels_dir = os.path.join(self.yolo_temp_dir, "labels", "val")

        for dir in [train_imgs_dir, val_imgs_dir, train_labels_dir, val_labels_dir]:
            os.makedirs(dir)

        # Get lists of label files
        all_labels = [
            os.path.split(label)[-1]
            for label in glob.glob(os.path.join(label_dir, "*.txt"))
        ]

        # Match imgs and labels based on filenames
        img_label_pairs = [
            (label.replace(".txt", ".png"), label) for label in all_labels
        ]

        # Shuffle the pairs randomly
        random.shuffle(img_label_pairs)

        # Calculate split sizes based on percentages
        split_index = int(len(img_label_pairs) * split_percentage)
        train_pairs = img_label_pairs[:split_index]
        val_pairs = img_label_pairs[split_index:]

        # Move train images and labels to train directories
        for img, label in train_pairs:
            src_img = os.path.join(img_dir, img)
            dest_img = os.path.join(train_imgs_dir, img)
            shutil.copy(src_img, dest_img)

            src_label = os.path.join(label_dir, label)
            dest_label = os.path.join(train_labels_dir, label)
            shutil.copy(src_label, dest_label)

        # Move test images and labels to test directories
        for img, label in val_pairs:
            src_img = os.path.join(img_dir, img)
            dest_img = os.path.join(val_imgs_dir, img)
            shutil.copy(src_img, dest_img)

            src_label = os.path.join(label_dir, label)
            dest_label = os.path.join(val_labels_dir, label)
            shutil.copy(src_label, dest_label)

    def load_weights(self, weights):
        self.model = YOLO(weights)

    def train_model(self, img_dir, submask_dir, weights_path):
        if self.temp_data_yaml is None:
            self.create_temp_yaml()

        self.model.train(
            data=self.temp_data_yaml,
            imgsz=(1024, 768),  # ToDo: Replace with passed in size arg
            device=0,
            epochs=100,
            flipud=0.5,
        )
        # ToDo: Copy weights file from ultralytics weights to datastore weights

    def get_map_value(self):
        if self.temp_data_yaml is None:
            self.create_temp_yaml()
        metrics = self.model.val(data=self.temp_data_yaml)
        print(f"mAP@75: {metrics.seg.map75}")

    def create_temp_yaml(self):
        # ToDo: Look into why these settings don't take effect until run twice
        settings.update(
            {
                "datasets_dir": "",
                "runs_dir": os.path.join(self.project_path, "ultralytics/runs"),
                "weights_dir": os.path.join(self.project_path, "ultralytics/weights"),
            }
        )
        if self.yolo_temp_dir is None:
            raise "Run the create_temp_split method before training"
        data = {
            "path": self.yolo_temp_dir,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.default_class_map(),
        }

        # Writing to YAML file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            yaml.dump(data, temp_file, default_flow_style=False)
            self.temp_data_yaml = temp_file.name
