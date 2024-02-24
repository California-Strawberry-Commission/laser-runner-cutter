import os
import argparse
from glob import glob
import cv2
import numpy as np
import torch
import torchvision
import torchmetrics
from torchvision.transforms import v2 as transforms
from time import perf_counter
from tqdm import tqdm
import albumentations as A

PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_PREPARED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/prepared/runner1800",
)
DEFAULT_SIZE = (1024, 768)
DEFAULT_EPOCHS = 150


class AlbumRandAugment:
    def __init__(self, basic_count=0, complex_count=0, size=DEFAULT_SIZE):
        self.basic_count = basic_count
        self.complex_count = complex_count
        self.size = size

    @property
    def basic_augmentations(self):
        return [
            None,
            A.RandomRotate90(),
            A.Transpose(),
        ]

    @property
    def complex_augmentations(self):
        return [
            None,
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.ElasticTransform(p=1),
            A.OpticalDistortion(p=1),
            A.Blur(p=1),
        ]

    def apply(self, image, masks=None):
        base_augments = np.random.choice(self.basic_augmentations, self.basic_count)
        base_augments = [aug for aug in base_augments if aug is not None]
        complex_augments = np.random.choice(
            self.complex_augmentations, self.complex_count
        )
        complex_augments = [aug for aug in complex_augments if aug is not None]

        transform = A.Compose(
            base_augments + complex_augments + [A.Resize(self.size[1], self.size[0])]
        )
        # Transform function will raise an error is masks are passed in with zero size
        if masks is not None and len(masks) > 0:
            return transform(image=image, masks=masks)
        else:
            return transform(image=image)


class MaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, size=DEFAULT_SIZE, transformer=None):
        """Custom dataset for Mask RCNN

        Args:
            img_dir (string): path to image directory
            mask_dir (string): path to mask directory. Note that this directory must
                                contain subdirectories with names matching the image
                                files in img_dir without the extension
            size ((width, height)): width and height of images and masks. Images and
                                    masks will be resized to this size.
            transformer: transform generator
        """
        self.img_paths = glob(os.path.join(img_dir, "*.jpg")) + glob(
            os.path.join(img_dir, "*.png")
        )
        self.mask_dir = mask_dir
        self.size = size
        self.transformer = transformer

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = self._get_masks_for_image(img_path)

        if self.transformer is not None:
            # Apply augmentations
            augmented = self.transformer.apply(img, masks)
            img = augmented["image"]
            masks = augmented.get("masks", [])

            # Augmentation can result in masks with no valid pixels, so filter them out
            valid_masks = []
            for mask in masks:
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                if np.max(mask) == 255:
                    valid_masks.append(mask)
            masks = np.array(valid_masks)

            # Filter out bboxes that have no width or height
            mask_tensor = torch.as_tensor(masks, dtype=torch.float32)
            bboxes = torchvision.ops.boxes.masks_to_boxes(mask_tensor)
            valid_masks = []
            valid_bboxes = []
            for mask, bbox in zip(masks, bboxes):
                if not (bbox[0] == bbox[2] or bbox[1] == bbox[3]):
                    valid_masks.append(mask)
                    valid_bboxes.append(bbox.numpy())
            masks = np.array(valid_masks)
            bboxes = np.array(valid_bboxes)
        else:
            mask_tensor = torch.as_tensor(masks, dtype=torch.float32)
            bboxes = torchvision.ops.boxes.masks_to_boxes(mask_tensor)

        # Create tensors for image and masks
        img_tensor = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )(img)
        # If there are no masks, create empty tensors of the correct size
        if len(masks) == 0:
            mask_tensor = torch.empty((0, self.size[1], self.size[0]), dtype=torch.bool)
            bbox_tensor = torch.empty((0, 4), dtype=torch.float32)
            label_tensor = torch.empty((0), dtype=torch.int64)
        else:
            # Convert masks from float nparray to boolean tensor
            mask_tensor = torch.as_tensor(masks > 0, dtype=torch.bool)
            bbox_tensor = torch.as_tensor(bboxes, dtype=torch.float32)
            label_tensor = torch.as_tensor(np.ones(len(masks)), dtype=torch.int64)

        return img_tensor, {
            "boxes": bbox_tensor,
            "labels": label_tensor,
            "masks": mask_tensor,
        }

    def _get_masks_for_image(self, img_path):
        mask_subdir = os.path.join(
            self.mask_dir, os.path.splitext(os.path.basename(img_path))[0]
        )
        mask_paths = glob(os.path.join(mask_subdir, "*.jpg")) + glob(
            os.path.join(mask_subdir, "*.png")
        )

        masks = np.zeros((0, self.size[1], self.size[0]))
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.size)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            # Don't include masks without any valid pixel values
            if np.max(mask) < 255:
                continue

            masks = np.concatenate((masks, mask[np.newaxis, :, :]), axis=0)

        return masks


class MaskRCNN:
    def __init__(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # Currently only working with one class
        num_classes = 2
        self.model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        )
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = (
            torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                in_features_mask, hidden_layer, num_classes
            )
        )
        self.model.to(self.device)

    def load_weights(self, weights_file):
        self.model.load_state_dict(torch.load(weights_file))
        self.model.eval()
        self.model.to(self.device)

    def train(
        self,
        images_dir,
        masks_dir,
        weights_file,
        size=DEFAULT_SIZE,
        epochs=DEFAULT_EPOCHS,
        verbose=False,
    ):
        train_data = torch.utils.data.DataLoader(
            MaskRCNNDataset(
                os.path.join(images_dir, "train"),
                os.path.join(masks_dir, "train"),
                size=size,
            ),
            batch_size=2,
            shuffle=True,
            collate_fn=lambda batch: batch,
            num_workers=12,
            pin_memory=torch.cuda.is_available(),
        )
        val_data = torch.utils.data.DataLoader(
            MaskRCNNDataset(
                os.path.join(images_dir, "val"),
                os.path.join(masks_dir, "val"),
                size=size,
            ),
            batch_size=2,
            shuffle=True,
            collate_fn=lambda batch: batch,
            num_workers=12,
            pin_memory=torch.cuda.is_available(),
        )

        # Construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=1e-4)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )

        min_val_loss = np.inf
        for epoch in tqdm(range(epochs)):
            epoch_start = perf_counter()
            train_loss_list = []
            val_loss_list = []
            self.model.train()
            for batch in tqdm(train_data):
                imgs = [img.to(self.device) for img, _ in batch]
                labels = [
                    {k: v.to(self.device) for k, v in label.items()}
                    for _, label in batch
                ]
                loss = self.model(imgs, labels)
                if verbose:
                    print(loss)
                losses = sum([l for l in loss.values()])
                train_loss_list.append(losses.cpu().detach().numpy())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            # Update the learning rate
            lr_scheduler.step()
            with torch.no_grad():
                for batch in val_data:
                    imgs = [img.to(self.device) for img, _ in batch]
                    labels = [
                        {k: v.to(self.device) for k, v in label.items()}
                        for _, label in batch
                    ]
                    loss = self.model(imgs, labels)
                    losses = sum([l for l in loss.values()])
                    val_loss_list.append(losses.cpu().detach().numpy())

            train_loss = np.average(np.array(train_loss_list))
            val_loss = np.average(np.array(val_loss_list))
            epoch_stop = perf_counter()
            print(
                f"Epoch {epoch} took {epoch_stop - epoch_start} seconds. Train loss = {train_loss}, val loss = {val_loss}"
            )
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                os.makedirs(os.path.dirname(weights_file), exist_ok=True)
                torch.save(self.model.state_dict(), weights_file)
                print(f"Saved weights file to {weights_file}")

    def eval(self, images_dir, masks_dir, size=DEFAULT_SIZE):
        test_data = torch.utils.data.DataLoader(
            MaskRCNNDataset(
                os.path.join(images_dir, "test"),
                os.path.join(masks_dir, "test"),
                size=size,
            ),
            batch_size=1,
            shuffle=True,
            collate_fn=lambda batch: batch,
            num_workers=12,
            pin_memory=torch.cuda.is_available(),
        )
        self.model.eval()
        with torch.no_grad():
            metric_bbox = torchmetrics.detection.MeanAveragePrecision(iou_type="bbox")
            metric_segm = torchmetrics.detection.MeanAveragePrecision(iou_type="segm")
            for batch in tqdm(test_data):
                images = [img.to(self.device) for img, _ in batch]
                labels = [
                    {k: v.to(self.device) for k, v in label.items()}
                    for _, label in batch
                ]
                preds = self.model(images)
                labels_cpu = []
                preds_cpu = []
                for label in labels:
                    d = {k: v.cpu().detach() for k, v in label.items()}
                    # For MeanAveragePrecision.update(), make sure pred["masks"] is a boolean tensor
                    # with the shape (n, h, w) where n is the number of masks, w is the width,
                    # and h is the height
                    d["masks"] = d["masks"] > 0
                    labels_cpu.append(d)
                for pred in preds:
                    d = {k: v.cpu().detach() for k, v in pred.items()}
                    # For MeanAveragePrecision.update(), make sure pred["masks"] is a boolean tensor
                    # with the shape (n, h, w) where n is the number of predicted masks, w is the width,
                    # and h is the height
                    d["masks"] = (d["masks"] > 0).squeeze(1)
                    preds_cpu.append(d)
                metric_bbox.update(preds_cpu, labels_cpu)
                metric_segm.update(preds_cpu, labels_cpu)

            return metric_bbox.compute(), metric_segm.compute()


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
        "--images_dir",
        default=os.path.join(DEFAULT_PREPARED_DATA_DIR, "images"),
    )
    train_parser.add_argument(
        "--masks_dir",
        default=os.path.join(DEFAULT_PREPARED_DATA_DIR, "masks"),
    )
    train_parser.add_argument(
        "--weights_file", default=os.path.join(PROJECT_PATH, "maskrcnn", "maskrcnn.pt")
    )
    train_parser.add_argument(
        "--size", type=tuple_type, default=f"({DEFAULT_SIZE[0]}, {DEFAULT_SIZE[1]})"
    )
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)

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
        "--weights_file", default=os.path.join(PROJECT_PATH, "maskrcnn", "maskrcnn.pt")
    )
    eval_parser.add_argument(
        "--size", type=tuple_type, default=f"({DEFAULT_SIZE[0]}, {DEFAULT_SIZE[1]})"
    )

    args = parser.parse_args()

    model = MaskRCNN()
    weights_file = args.weights_file
    if os.path.exists(weights_file):
        model.load_weights(weights_file)

    if args.command == "train":
        model.train(
            args.images_dir,
            args.masks_dir,
            weights_file,
            size=args.size,
            epochs=args.epochs,
        )
    elif args.command == "eval":
        metric_bbox, metric_segm = model.eval(
            args.images_dir, args.masks_dir, size=args.size
        )
        print(metric_bbox)
        print(metric_segm)
    else:
        print("Invalid command.")
