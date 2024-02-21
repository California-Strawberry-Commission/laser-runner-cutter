import os
from glob import glob
import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2 as transforms
from time import perf_counter


class MaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, size=(1024, 768)):
        """Custom dataset for Mask RCNN

        Args:
            img_dir (string): path to image directory
            mask_dir (string): path to mask directory. Note that this directory must
                                contain subdirectories with names matching the image
                                files in img_dir without the extension
            size ((width, height)): width and height of images and masks. Images and
                                    masks will be resized to this size.
        """
        self.img_paths = glob(os.path.join(img_dir, "*.jpg")) + glob(
            os.path.join(img_dir, "*.png")
        )
        self.mask_dir = mask_dir
        self.size = size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )(img)

        masks = self._get_masks_for_image(img_path)
        mask_tensor = torch.as_tensor(masks, dtype=torch.float32)
        label_tensor = torch.as_tensor(np.ones(len(masks)), dtype=torch.int64)
        bbox_tensor = torchvision.ops.boxes.masks_to_boxes(mask_tensor)
        data = {
            "boxes": bbox_tensor,
            "labels": label_tensor,
            "masks": mask_tensor,
        }

        return img_tensor, data

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
            if np.max(mask) < 255:
                continue

            masks = np.concatenate((masks, mask[np.newaxis, :, :]), axis=0)

        return masks


def collate_fn(batch):
    return batch


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
        self.model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
        )
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = (
            torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                in_features_mask, hidden_layer, 2
            )
        )
        self.model.to(self.device)

    def train(self, weights_path, verbose=False):
        train_data = torch.utils.data.DataLoader(
            MaskRCNNDataset(
                "/home/genki/ros2_ws/src/laser-runner-cutter/runner_segmentation_model/data/prepared/runner1800/images/train",
                "/home/genki/ros2_ws/src/laser-runner-cutter/runner_segmentation_model/data/prepared/runner1800/masks/train",
            ),
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=12,
            pin_memory=torch.cuda.is_available(),
        )
        val_data = torch.utils.data.DataLoader(
            MaskRCNNDataset(
                "/home/genki/ros2_ws/src/laser-runner-cutter/runner_segmentation_model/data/prepared/runner1800/images/val",
                "/home/genki/ros2_ws/src/laser-runner-cutter/runner_segmentation_model/data/prepared/runner1800/masks/val",
            ),
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
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
        for epoch in range(100):
            epoch_start = perf_counter()
            train_loss_list = []
            val_loss_list = []
            self.model.train()
            for dt in train_data:
                imgs = [img.to(self.device) for img, _ in dt]
                labels = [
                    {k: v.to(self.device) for k, v in label.items()} for _, label in dt
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
                for dt in val_data:
                    imgs = [img.to(self.device) for img, _ in dt]
                    labels = [
                        {k: v.to(self.device) for k, v in label.items()}
                        for _, label in dt
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
                torch.save(self.model.state_dict(), weights_path)
                print(f"Saved weights file to {weights_path}")


project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
model = MaskRCNN()
model.train(os.path.join(project_path, "models", "runner1800-maskrcnn.pt"))
