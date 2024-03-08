import os
import argparse
from glob import glob
import cv2
import numpy as np
import torch
import torchvision
import torchmetrics
from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as F
from time import perf_counter
from tqdm import tqdm
import albumentations as A
from functools import partial
import matplotlib
import matplotlib.pyplot as plt

PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_PREPARED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/prepared/runner1800",
)
DEFAULT_SIZE = (1024, 768)
DEFAULT_EPOCHS = 150

matplotlib.use("tkagg")


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
    def __init__(self, size=DEFAULT_SIZE):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Initialize a Mask R-CNN model with pretrained weights
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT"
        )

        # We need to replace the bounding box and segmentation mask predictors for the
        # pretrained model with new ones for our dataset.
        num_classes = 2  # 1 class (runner) + background
        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features_box, num_classes
            )
        )
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = self.model.roi_heads.mask_predictor.conv5_mask.out_channels
        self.model.roi_heads.mask_predictor = (
            torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                in_features_mask, dim_reduced, num_classes
            )
        )

        self.model.to(self.device)

        self.size = size

    def load_weights(self, weights_file):
        self.model.load_state_dict(torch.load(weights_file))
        self.model.eval()
        self.model.to(self.device)

    def train(
        self,
        images_dir,
        masks_dir,
        weights_file,
        epochs=DEFAULT_EPOCHS,
        verbose=False,
    ):
        train_data = torch.utils.data.DataLoader(
            MaskRCNNDataset(
                os.path.join(images_dir, "train"),
                os.path.join(masks_dir, "train"),
                size=self.size,
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
                size=self.size,
            ),
            batch_size=2,
            shuffle=True,
            collate_fn=lambda batch: batch,
            num_workers=12,
            pin_memory=torch.cuda.is_available(),
        )

        # Construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        lr = 5e-4
        optimizer = torch.optim.AdamW(
            params, lr=lr
        )  # AdamW optimizer includes weight decay for regularization
        # and a learning rate scheduler
        """
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )
        """
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=epochs * len(train_data)
        )

        # Initialize a gradient scaler for mixed-precision training if the device is a CUDA GPU
        scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None

        self.model.train()
        best_val_loss = float("inf")
        for epoch in tqdm(range(epochs), desc="Train"):
            epoch_start = perf_counter()
            train_loss_list = []
            val_loss_list = []
            for batch in tqdm(train_data, desc="Epoch"):
                imgs = [img.to(self.device) for img, _ in batch]
                labels = [
                    {k: v.to(self.device) for k, v in label.items()}
                    for _, label in batch
                ]

                # Clear gradients, as PyTorch accumulates gradients by default
                optimizer.zero_grad()

                # Forward pass with Automatic Mixed Precision (AMP) context manager
                with torch.amp.autocast(torch.device(self.device).type):
                    loss_dict = self.model(imgs, labels)
                    if verbose:
                        print(loss_dict)
                    loss = sum([loss for loss in loss_dict.values()])
                    train_loss_list.append(loss.item())

                # Backpropagate the error and update the weights
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    old_scaler = scaler.get_scale()
                    scaler.update()
                    new_scaler = scaler.get_scale()
                    if new_scaler >= old_scaler:
                        lr_scheduler.step()
                else:
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

            # Get the validation loss
            with torch.no_grad():
                for batch in val_data:
                    imgs = [img.to(self.device) for img, _ in batch]
                    labels = [
                        {k: v.to(self.device) for k, v in label.items()}
                        for _, label in batch
                    ]

                    # Forward pass with Automatic Mixed Precision (AMP) context manager
                    with torch.amp.autocast(torch.device(self.device).type):
                        loss_dict = self.model(imgs, labels)
                        loss = sum([loss for loss in loss_dict.values()])
                        val_loss_list.append(loss.item())

            train_loss = np.average(np.array(train_loss_list))
            val_loss = np.average(np.array(val_loss_list))
            epoch_stop = perf_counter()
            print(
                f"Epoch {epoch} took {epoch_stop - epoch_start} seconds. Train loss = {train_loss}, val loss = {val_loss}"
            )

            # Save the model if it's the best validation loss we've seen
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(weights_file), exist_ok=True)
                torch.save(self.model.state_dict(), weights_file)
                print(f"Saved weights file to {weights_file}")

    def eval(self, images_dir, masks_dir):
        test_data = torch.utils.data.DataLoader(
            MaskRCNNDataset(
                os.path.join(images_dir, "test"),
                os.path.join(masks_dir, "test"),
                size=self.size,
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

    def debug(self, image_file, mask_subdir=None, conf_threshold=0.5):
        orig_img = cv2.imread(image_file)
        img = cv2.resize(orig_img, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )(img)[None].to(self.device)
        self.model.eval()
        with torch.no_grad():
            result = self.model(img_tensor)

        result = move_data_to_device(result, "cpu")[0]

        # Filter the output based on the confidence threshold
        conf_mask = result["scores"] > conf_threshold

        # Confidence scores
        conf = result["scores"][conf_mask]

        # Calculate the scale between the source image and the resized image
        min_img_scale = min(orig_img.shape[:2]) / min(self.size)
        # Scale the predicted bounding boxes to the source image
        pred_bboxes = torchvision.tv_tensors.BoundingBoxes(
            result["boxes"][conf_mask] * min_img_scale,
            format="xyxy",
            canvas_size=(self.size[1], self.size[0]),
        )

        # Scale and stack the predicted segmentation masks
        pred_masks = torch.nn.functional.interpolate(
            result["masks"][conf_mask], size=orig_img.shape[:2]
        )
        pred_masks = torch.concat(
            [
                torchvision.tv_tensors.Mask(
                    torch.where(mask >= conf_threshold, 1, 0), dtype=torch.bool
                )
                for mask in pred_masks
            ]
        )

        # Annotate and display image with ground truth masks and predicted masks
        def show(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]
            fix, axes = plt.subplots(ncols=len(imgs), squeeze=False)
            for i, img in enumerate(imgs):
                img = img.detach()
                img = F.to_pil_image(img)
                axes[0, i].imshow(np.asarray(img))
                axes[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.show()

        img_tensor = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.uint8, scale=True)]
        )(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        draw_bboxes = partial(
            torchvision.utils.draw_bounding_boxes, fill=False, width=2
        )

        images_to_display = []
        if mask_subdir is not None:
            mask_paths = glob(os.path.join(mask_subdir, "*.jpg")) + glob(
                os.path.join(mask_subdir, "*.png")
            )
            masks = np.zeros((0, orig_img.shape[0], orig_img.shape[1]))
            for mask_path in mask_paths:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]))
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                # Don't include masks without any valid pixel values
                if np.max(mask) < 255:
                    continue

                masks = np.concatenate((masks, mask[np.newaxis, :, :]), axis=0)
            if len(masks) == 0:
                gt_masks = torchvision.tv_tensors.Mask(
                    torch.empty(
                        (0, orig_img.shape[0], orig_img.shape[1]), dtype=torch.bool
                    )
                )
            else:
                gt_masks = torchvision.tv_tensors.Mask(masks > 0, dtype=torch.bool)

            gt_bboxes = torchvision.tv_tensors.BoundingBoxes(
                data=torchvision.ops.masks_to_boxes(gt_masks),
                format="xyxy",
                canvas_size=orig_img.shape[:2],
            )

            # Annotate the image with the GT segmentation masks
            gt_annotated_tensor = torchvision.utils.draw_segmentation_masks(
                image=img_tensor, masks=gt_masks, alpha=0.3, colors="red"
            )
            # Annotate the image with the GT bounding boxes
            gt_annotated_tensor = draw_bboxes(
                image=gt_annotated_tensor,
                boxes=gt_bboxes,
                labels=["runner" for _ in range(gt_bboxes.size()[0])],
                colors="red",
            )
            images_to_display.append(gt_annotated_tensor)

        # Annotate the image with the predicted segmentation masks
        pred_annotated_tensor = torchvision.utils.draw_segmentation_masks(
            image=img_tensor, masks=pred_masks, alpha=0.3, colors="red"
        )
        # Annotate the image with the predicted labels and bounding boxes
        pred_annotated_tensor = draw_bboxes(
            image=pred_annotated_tensor,
            boxes=pred_bboxes,
            labels=[f"runner: {c:.2f}" for c in conf],
            colors="red",
        )
        images_to_display.append(pred_annotated_tensor)

        # Display GT (if available) and predictions
        show(images_to_display)


def move_data_to_device(
    data,
    device: torch.device,
):
    """
    Recursively move data to the specified device.

    This function takes a data structure (could be a tensor, list, tuple, or dictionary)
    and moves all tensors within the structure to the given PyTorch device.

    Args:
    data (any): data to move to the device.
    device (torch.device): the PyTorch device to move the data to.
    """

    # If the data is a tuple, iterate through its elements and move each to the device.
    if isinstance(data, tuple):
        return tuple(move_data_to_device(d, device) for d in data)

    # If the data is a list, iterate through its elements and move each to the device.
    if isinstance(data, list):
        return list(move_data_to_device(d, device) for d in data)

    # If the data is a dictionary, iterate through its key-value pairs and move each value to the device.
    elif isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}

    # If the data is a tensor, directly move it to the device.
    elif isinstance(data, torch.Tensor):
        return data.to(device)

    # If the data type is not a tensor, list, tuple, or dictionary, it remains unchanged.
    else:
        return data


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
        "--weights_file",
        default=os.path.join(PROJECT_PATH, "output", "maskrcnn", "maskrcnn.pt"),
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
        "--weights_file",
        default=os.path.join(PROJECT_PATH, "output", "maskrcnn", "maskrcnn.pt"),
    )
    eval_parser.add_argument(
        "--size", type=tuple_type, default=f"({DEFAULT_SIZE[0]}, {DEFAULT_SIZE[1]})"
    )

    debug_parser = subparsers.add_parser("debug", help="Debug model predictions")
    debug_parser.add_argument(
        "--weights_file",
        default=os.path.join(PROJECT_PATH, "output", "maskrcnn", "maskrcnn.pt"),
    )
    debug_parser.add_argument("--image_file", required=True)
    debug_parser.add_argument("--mask_subdir")
    debug_parser.add_argument(
        "--size", type=tuple_type, default=f"({DEFAULT_SIZE[0]}, {DEFAULT_SIZE[1]})"
    )

    args = parser.parse_args()

    model = MaskRCNN(size=args.size)
    weights_file = args.weights_file
    if weights_file is not None and os.path.exists(weights_file):
        model.load_weights(weights_file)

    if args.command == "train":
        model.train(
            args.images_dir,
            args.masks_dir,
            weights_file,
            epochs=args.epochs,
        )
    elif args.command == "eval":
        metric_bbox, metric_segm = model.eval(args.images_dir, args.masks_dir)
        print(metric_bbox)
        print(metric_segm)
    elif args.command == "debug":
        model.debug(args.image_file, args.mask_subdir)
    else:
        print("Invalid command.")
