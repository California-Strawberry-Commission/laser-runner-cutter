import asyncio
import functools
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from runner_segmentation.yolo import Yolo


class LaserDetector:
    def __init__(self, logger: Optional[logging.Logger] = None):
        models_dir = os.path.join(
            get_package_share_directory("camera_control"),
            "models",
        )
        model_weights_path = os.path.join(models_dir, "LaserDetectionYoloV8n.pt")
        self._input_image_size = (1024, 768)
        self._model = Yolo(
            weights_file=model_weights_path,
            input_image_size=self._input_image_size,
        )
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)

    async def detect(
        self, color_frame: np.ndarray, conf_threshold: float = 0.0
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Detects laser points in a color image.

        Args:
            color_frame (np.ndarray): Color image.
            conf_threshold (float): Minimum confidence score that should be considered.
        Returns:
            Tuple[List[Tuple[int, int]], List[float]]: A tuple of (list of detected laser points, list of associated confidence scores).
        """
        # Scale image before prediction to improve performance
        frame_width = color_frame.shape[1]
        frame_height = color_frame.shape[0]
        result_width = self._input_image_size[0]
        result_height = self._input_image_size[1]
        color_frame = cv2.resize(
            color_frame,
            self._input_image_size,
            interpolation=cv2.INTER_LINEAR,
        )

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                self._model.predict,
                color_frame,
            ),
        )
        result_conf = result["conf"]
        self._logger.debug(f"Laser point prediction found {result_conf.size} objects.")

        laser_points = []
        confs = []
        for idx in range(result["conf"].size):
            conf = float(result["conf"][idx])
            if conf >= conf_threshold:
                # bbox is in xyxy format
                bbox = result["bboxes"][idx]
                # Scale the result coords to frame coords
                laser_points.append(
                    (
                        round((bbox[0] + bbox[2]) * 0.5 * frame_width / result_width),
                        round((bbox[1] + bbox[3]) * 0.5 * frame_height / result_height),
                    )
                )
                confs.append(conf)
            else:
                self._logger.info(
                    f"Laser point prediction ignored due to low confidence: {conf} < {conf_threshold}"
                )
        return laser_points, confs
