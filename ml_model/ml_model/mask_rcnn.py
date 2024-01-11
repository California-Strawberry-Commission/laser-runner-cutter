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
from tqdm import tqdm
import random
import albumentations as A
from ml_model.model_base import ModelBase
from ml_model.utils import find_closest_point
from torchmetrics.detection import MeanAveragePrecision

SIZE = (768, 1024)

class AlbumRandAugment:
    def __init__(self, basic_count=0, complex_count=0):
        self.basic_count = basic_count
        self.complex_count = complex_count

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

    def apply(self, image, mask=None, masks=None):
        # Use choice instead of sample
        base_augments = np.random.choice(self.basic_augmentations, self.basic_count)
        base_augments = [aug for aug in base_augments if aug is not None]
        complex_augments = np.random.choice(
            self.complex_augmentations, self.complex_count
        )
        complex_augments = [aug for aug in complex_augments if aug is not None]

        transform = A.Compose(
            base_augments + complex_augments + [A.Resize(SIZE[0], SIZE[1])]
        )
        # Transform function will raise an error is masks are passed in with zero size
        if masks is not None and len(masks) > 0:
            return transform(image=image, masks=masks)
        elif mask is not None and len(mask) > 0:
            return transform(image=image, mask=mask)
        else:
            return transform(image=image)


class MRCNNDataSet(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_subdirs, transform_gen=None):
        # list of image paths
        self.img_paths = image_paths
        # list of mask subdirectories
        self.mask_subdirs = mask_subdirs

        self.transform_gen = transform_gen

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks, labels = self._find_mrcnn_labels(idx)
        augmented = self.transform_gen.apply(img, masks=masks)
        img_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(
            augmented["image"]
        )

        included_masks = []
        included_labels = []
        aug_masks = augmented.get("masks", [])
        for label, mask in zip(labels, aug_masks):
            # Augmentation can cause masks with no valid pixels causes errors for masks
            ret, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
            if np.max(mask) == 255:
                included_masks.append(mask)
                included_labels.append(label)

        included_masks = np.array(included_masks) / 255
        mask_tensor = torch.from_numpy(included_masks)
        bboxes = masks_to_boxes(mask_tensor)

        inc2_label = []
        inc2_masks = []
        inc2_bboxes = []
        # Remove bboxes that are single pixel length or width
        for label, mask, bbox in zip(included_labels, included_masks, bboxes):
            if not (bbox[0] == bbox[2] or bbox[1] == bbox[3]):
                inc2_label.append(label)
                inc2_masks.append(mask)
                inc2_bboxes.append(bbox.numpy())
        # If the masks are empty, create empty tensors of ther correct size
        if len(inc2_masks) == 0:
            mask_tensor = torch.empty((0, SIZE[0], SIZE[1]), dtype=torch.float32)
            bbox_tensor = torch.empty((0, 4), dtype=torch.float32)
            label_tensor = torch.empty((0), dtype=torch.int64)
        else:
            mask_tensor = torch.as_tensor(np.array(inc2_masks), dtype=torch.float32)
            bbox_tensor = torch.as_tensor(np.array(inc2_bboxes), dtype=torch.float32)
            label_tensor = torch.as_tensor(np.array(inc2_label), dtype=torch.int64)

        data = {
            "boxes": bbox_tensor,
            "labels": label_tensor,
            "masks": mask_tensor,
        }

        return img_tensor, data

    def __len__(self):
        return len(self.img_paths)

    def _find_mrcnn_labels(self, idx):
        mask_subdir = self.mask_subdirs[idx]
        paths = glob.glob(os.path.join(mask_subdir, "*"))

        # This doesn't work for more then one class type, add the class id to the subdir name?
        labels = np.ones(len(paths))
        masks = np.zeros((len(paths), SIZE[0], SIZE[1]))
        for idx, mask_path in enumerate(paths):
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, (SIZE[1], SIZE[0]))
            masks[idx] = mask[:, :, 0]
            labels[idx] = 1

        return masks, labels


def collate_fn(batch):
    return batch


class MaskRCNN(ModelBase):
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

    def inference_transform(self, img_arr):
        return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(img_arr)

    def get_centroids(self, img_arr, score_thresh=0.25, mask_thresh=0.2):
        """Return a list of centroids for each detection.
                Currently does not support batches

        Args:
            img_arr (ndarray[H, W, 3]): RGB ndarray representing the image.

        Returns:
            [(float, float), .]: List of X, Y points
        """
        img_tensor = self.inference_transform(img_arr)
        img_tensor = img_tensor.to(self.device)
        res = self.model([img_tensor])
        point_list = []
        score_list = []
        if len(res[0]["masks"]):
            for score, mask_tensor in zip(res[0]["scores"], res[0]["masks"]):
                if score < score_thresh:
                    # Could return here is scores are sorted
                    continue
                mask_img = mask_tensor.detach().cpu().numpy()
                mask_img = mask_img[0, :, :]
                ret, mask_img = cv2.threshold(
                    mask_img, mask_thresh, 1, cv2.THRESH_BINARY
                )
                y_c, x_c = np.argwhere(mask_img == 1).sum(0) / np.count_nonzero(
                    mask_img
                )
                closest_point = find_closest_point(mask_img, (x_c, y_c))
                # if not np.isnan(x_c) and np.isnan()
                point_list.append(closest_point)
                score_list.append(score)

        return score_list, point_list

    def get_map_value(self, img_dir, mask_dir):
        subdir_paths = glob.glob(os.path.join(mask_dir, "*"))
        imgs = [
            os.path.join(img_dir, os.path.split(mask_subdir)[-1] + ".png")
            for mask_subdir in subdir_paths
        ]
        # Using the dataloader concept to create the test labels
        test_dl = torch.utils.data.DataLoader(
            MRCNNDataSet(
                imgs,
                subdir_paths,
                AlbumRandAugment(basic_count=0, complex_count=0),
            ),
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=12,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        self.model.eval()
        metric = MeanAveragePrecision()
        for dt in tqdm(test_dl):
            labels_set = []
            pred_set = []
            imgs = [img.to(self.device) for img, _ in dt]
            labels = [
                {k: v.to(self.device) for k, v in label.items()} for _, label in dt
            ]
            preds = self.model(imgs)
            for label in labels:
                labels_set.append({k: v.cpu().detach() for k, v in label.items()})
            for pred in preds:
                pred_set.append({k: v.cpu().detach() for k, v in pred.items()})
            metric.update(pred_set, labels_set)

        met = metric.compute()
        return met

    def load_weights(self, load_path):
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
            os.path.join(img_dir, os.path.split(mask_subdir)[-1] + ".png")
            for mask_subdir in train_mask_subdirs
        ]
        val_imgs = [
            os.path.join(img_dir, os.path.split(mask_subdir)[-1] + ".png")
            for mask_subdir in val_mask_subdirs
        ]

        train_dl = torch.utils.data.DataLoader(
            MRCNNDataSet(
                train_imgs,
                train_mask_subdirs,
                AlbumRandAugment(basic_count=2, complex_count=2),
            ),
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=12,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        val_dl = torch.utils.data.DataLoader(
            MRCNNDataSet(
                val_imgs,
                val_mask_subdirs,
                AlbumRandAugment(basic_count=0, complex_count=0),
            ),
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=12,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=1e-4)
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
                print(f"saving weights file to {weights_path}")
                torch.save(self.model.state_dict(), weights_path)

    @staticmethod
    def name():
        return "torch_mask_rcnn"
