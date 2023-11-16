import numpy as np
import glob
import os
import cv2


def split_mask(mask_dir, mask_subdir):
    mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path)
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        mask = mask[:, :, 0]
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        mask_name = os.path.splitext(os.path.split(mask_path)[-1])[0]
        new_subdir = os.path.join(mask_subdir, mask_name)

        os.mkdir(new_subdir)
        for idx in range(len(contours)):
            new_mask = np.zeros(mask.shape)
            cv2.drawContours(
                new_mask, contours, idx, color=(255, 255, 255), thickness=-1
            )
            new_path = os.path.join(new_subdir, f"{idx}.jpg")
            cv2.imwrite(new_path, new_mask)
