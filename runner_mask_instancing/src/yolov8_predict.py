import os
import torch.nn.functional as F
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
from segment_utils import convert_mask_to_line_segments


DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/raw",
)
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../models",
)
MODEL_PATH = os.path.join(MODELS_DIR, "yolov8-segment", "weights", "best.pt")


model = YOLO(MODEL_PATH)
results = model.predict(
    os.path.join(DATA_DIR, "runner500", "151.png"),
    imgsz=(1024, 768),
    iou=0.5,
    conf=0.3,
)
for result in results:
    plot_array = result.plot()
    plot_image = Image.fromarray(plot_array[..., ::-1])
    plot_image.show()

    if result.masks is None:
        continue

    masks_data = result.masks.data
    # Resize masks to original image size
    masks_data = F.interpolate(
        masks_data.unsqueeze(1),
        size=(result.orig_img.shape[0], result.orig_img.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    masks_data = masks_data.squeeze(1)

    masks_data[masks_data != 0] = 255
    masks_np = masks_data.byte().cpu().numpy()

    confidences_np = result.boxes.conf.cpu().numpy()
    print(f"Confidences: {confidences_np}")

    for i in range(len(masks_np)):
        mask = masks_np[i]

        # Remove small contours from mask
        area_threshold = 64
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold
        ]
        filtered_mask = np.zeros_like(mask)
        cv2.drawContours(
            filtered_mask, filtered_contours, -1, 255, thickness=cv2.FILLED
        )

        im = Image.fromarray(filtered_mask)
        draw = ImageDraw.Draw(im)

        points = convert_mask_to_line_segments(filtered_mask, 4.0)
        if len(points) < 2:
            continue

        for i in range(len(points) - 1):
            draw.line(
                [(points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1])],
                fill=128,
                width=2,
            )

        im.show()
