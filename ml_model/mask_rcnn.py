import numpy as np
import os
import cv2
import glob

import torch
from torchvision.io import ImageReadMode
import torchvision
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.boxes import masks_to_boxes
from torchvision.utils import save_image
from torchvision.transforms import v2 as T
import random

SIZE = (720, 960)


# Currently does not support taking in images and masks, so random actions introduce error.
def get_transform(h_chance, v_chance):
    transforms = []
    transforms.append(T.Resize(size=SIZE, antialias=True))
    if h_chance > 0.5:
        transforms.append(T.RandomHorizontalFlip(1))
    if v_chance > 0.5:
        transforms.append(T.RandomVerticalFlip(1))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


class MRCNNDataSet(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_subdirs, transform_gen=None):
        # list of image paths
        self.img_paths = image_paths
        # list of mask subdirectories
        self.mask_subdirs = mask_subdirs

        self.transform_gen = transform_gen

    def __getitem__(self, idx):
        img = read_image(self.img_paths[idx])
        img = img

        if self.transform_gen is not None:
            h_chance = random.random()
            v_chance = random.random()
            t_func = self.transform_gen(h_chance, v_chance)
            img = t_func(img)
        else:
            t_func = None

        data = self._find_mrcnn_labels(idx, t_func)

        return img, data

    def __len__(self):
        return len(self.img_paths)

    def _find_mrcnn_labels(self, idx, t_func, debug_dir=False):
        mask_subdir = self.mask_subdirs[idx]
        paths = glob.glob(os.path.join(mask_subdir, "*.jpg"))

        # This doesn't work for more then one class type, add the class id to the subdir name?
        labels = np.ones(len(paths))
        # Try to move to all tensor actions in the
        masks = torch.zeros((len(paths), SIZE[0], SIZE[1]))
        for idx, mask_path in enumerate(paths):
            mask = read_image(mask_path, mode=ImageReadMode.UNCHANGED)
            mask = t_func(mask)
            masks[idx] = mask
            labels[idx] = 1

            if debug_dir:
                debug_img = cv2.rectangle(
                    mask,
                    (int(bboxes[idx][0]), int(bboxes[idx][1])),
                    (int(bboxes[idx][2]), int(bboxes[idx][3])),
                    color=(0, 0, 255),
                )
                cv2.imwrite(debug_dir, debug_img)

        labels = torch.as_tensor(labels, dtype=torch.int64)
        bboxes = masks_to_boxes(masks)
        return {
            "boxes": bboxes,
            "labels": labels,
            "masks": masks,
        }


def collate_fn(batch):
    return batch


class MaskRCNN:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # Currently only working with on class
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, 2
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        self.model.eval()
        self.model.to(self.device)

    def train_model(self, img_dir, mask_dir, weights_path, verbose=False):
        """Masks set the train/validate split"""
        subdir_paths = glob.glob(os.path.join(mask_dir, "*"))
        training_num = int(0.8 * len(subdir_paths))
        t_img_idx = np.random.choice(
            range(len(subdir_paths)), training_num, replace=False
        )
        v_img_idx = np.setdiff1d(range(len(subdir_paths)), t_img_idx)
        train_mask_subdirs = np.array(subdir_paths)[t_img_idx]
        val_mask_subdirs = np.array(subdir_paths)[v_img_idx]
        train_imgs = [
            os.path.join(img_dir, os.path.split(mask_subdir)[-1] + ".jpg")
            for mask_subdir in train_mask_subdirs
        ]
        val_imgs = [
            os.path.join(img_dir, os.path.split(mask_subdir)[-1] + ".jpg")
            for mask_subdir in val_mask_subdirs
        ]

        train_dl = torch.utils.data.DataLoader(
            MRCNNDataSet(train_imgs, train_mask_subdirs, get_transform),
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=16,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        val_dl = torch.utils.data.DataLoader(
            MRCNNDataSet(val_imgs, val_mask_subdirs, get_transform),
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=16,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )

        min_val_loss = np.inf
        for epoch in range(300):
            train_loss_list = []
            val_loss_list = []
            self.model.train()
            for dt in train_dl:
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

            # update the learning rate
            lr_scheduler.step()
            with torch.no_grad():
                for dt in val_dl:
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
            print(epoch, "  ", train_loss, "  ", val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), weights_path)


if __name__ == "__main__":
    weights_file = "runner_instance_segment.pt"
    model = MaskRCNN()
    weights_path = os.path.join(
        os.getcwd(), f"ml_model/data_store/weights/{weights_file}"
    )
    model.load(weights_path)
    img_dir = os.path.join(os.getcwd(), "ml_model/data_store/segmentation_data/raw/")
    mask_dir = os.path.join(
        os.getcwd(), "ml_model/data_store/segmentation_data/mrcnn_masks/"
    )
    model.train_model(img_dir, mask_dir, weights_path)
