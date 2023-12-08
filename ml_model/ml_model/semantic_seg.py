import numpy as np
import os
import cv2
import glob

import torch
from torchvision.io import read_image
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torchvision.utils import save_image
from torchvision.transforms import v2 as T

import random

from ml_model.model_base import ModelBase

SIZE = (768, 1024)


# Currently does not support taking in images and masks, so random actions introduce error.
def get_transform(h_chance, v_chance):
    transforms = []
    if h_chance > 0.5:
        transforms.append(T.RandomHorizontalFlip(1))
    if v_chance > 0.5:
        transforms.append(T.RandomVerticalFlip(1))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


class SemanticSegDataSet(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_subdirs, transform_gen=None):
        # list of image paths
        self.img_paths = image_paths
        # list of mask subdirectories
        self.mask_subdirs = mask_subdirs
        self.transform_gen = transform_gen

    def __getitem__(self, idx):
        img = read_image(self.img_paths[idx])
        img = T.Resize(size=SIZE, antialias=True)(img)
        masks = self._find_semantic_seg_mask(idx)

        if self.transform_gen is not None:
            h_chance = random.random()
            v_chance = random.random()
            t_func = self.transform_gen(h_chance, v_chance)
            img = t_func(img)
            masks = t_func(masks)
        return img, masks

    def __len__(self):
        return len(self.img_paths)

    def _find_semantic_seg_mask(self, idx, debug=False):
        mask_subdir = self.mask_subdirs[idx]
        paths = glob.glob(os.path.join(mask_subdir, "*.jpg"))

        # To do, move away from cv2 and into pure tensor implementation
        masks = np.zeros((1, SIZE[0], SIZE[1]), dtype=np.uint8)

        for idx, mask_path in enumerate(paths):
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, (SIZE[1], SIZE[0]))
            masks[0] = cv2.bitwise_or(masks[0], mask[:, :, 0])

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        return masks


def collate_fn(batch):
    return batch


class SemanticSegmentation(ModelBase):
    def __init__(self):
        self.model = models.segmentation.deeplabv3_resnet101(
            pretrained=True, progress=True
        )
        self.model.classifier = DeepLabHead(2048, 1)
        # Set the model in training mode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def load_weights(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def train_model(self, img_dir, mask_dir):
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
            SemanticSegDataSet(
                train_imgs, train_mask_subdirs, transform_gen=get_transform
            ),
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        val_dl = torch.utils.data.DataLoader(
            SemanticSegDataSet(val_imgs, val_mask_subdirs, transform_gen=get_transform),
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        criterion = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        min_val_loss = np.inf
        self.model.train()
        for epoch in range(300):
            train_loss_list = []
            val_loss_list = []
            for train_batch in train_dl:
                imgs = torch.zeros((len(train_batch), 3, SIZE[0], SIZE[1]))
                labels = torch.zeros((len(train_batch), 1, SIZE[0], SIZE[1]))

                for idx, sample in enumerate(train_batch):
                    imgs[idx] = sample[0]
                    labels[idx] = sample[1]
                    # save_image(sample[0], "")
                    # save_image(sample[1], "")

                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                y_pred = self.model(imgs)
                loss = criterion(y_pred["out"], labels)
                train_loss_list.append(loss.detach().numpy())
                print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for val_batch in val_dl:
                    imgs = torch.zeros((len(val_batch), 3, SIZE[0], SIZE[1]))
                    labels = torch.zeros((len(val_batch), 1, SIZE[0], SIZE[1]))
                    for idx, sample in enumerate(val_batch):
                        imgs[idx] = sample[0]
                        labels[idx] = sample[1]
                    loss = criterion(y_pred["out"], labels)
                    val_loss_list.append(loss.detach().numpy())

            train_loss = np.average(np.array(train_loss_list))
            val_loss = np.average(np.array(val_loss_list))
            print(epoch, "  ", train_loss, "  ", val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print("Saved new weights")
                torch.save(self.model.state_dict(), "")

    @staticmethod
    def name():
        return "torch_segmentation"
