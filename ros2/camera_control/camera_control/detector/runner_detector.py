import asyncio
import functools
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from ml_utils.mask_center import contour_center
from runner_segmentation.yolo import Yolo


class RunnerDetector:
    def __init__(self, logger: Optional[logging.Logger] = None):
        models_dir = os.path.join(
            get_package_share_directory("camera_control"),
            "models",
        )
        model_weights_path = os.path.join(models_dir, "RunnerSegYoloV8l.engine")
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
        self,
        color_frame: np.ndarray,
        conf_threshold: float = 0.0,
        bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[
        List[np.ndarray], List[Optional[Tuple[int, int]]], List[float], List[int]
    ]:
        """
        Detects runners in a color image. For each detected runner, a representative point is
        calculated. This point is the point on the runner that is closest to its centroid. If an
        optional `bounds` is defined, only that region of the mask will be considered when
        calculating the representative point.

        Args:
            color_frame (np.ndarray): Color image.
            conf_threshold (float): Minimum confidence score that should be considered.
            bounds (Tuple[int, int, int, int]): (min x, min y, width, height) to be considered when finding the representative point for each mask.
        Returns:
            Tuple[List[np.ndarray], List[Optional[Tuple[int, int]]], List[float], List[int]]: A tuple of (list of detected runner masks, list of representative point for each mask, list of associated confidence scores, list of associated instance IDs).
        """
        masks, confs, track_ids = await self._get_runner_masks(
            color_frame, conf_threshold
        )
        representative_points = await self._get_runner_representative_points(
            masks, bounds
        )
        return masks, representative_points, confs, track_ids

    async def _get_runner_masks(
        self, color_frame: np.ndarray, conf_threshold: float = 0.0
    ) -> Tuple[List[np.ndarray], List[float], List[int]]:
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
                self._model.track,
                color_frame,
            ),
        )
        result_conf = result["conf"]
        self._logger.debug(f"Runner mask prediction found {result_conf.size} objects.")

        runner_masks = []
        confs = []
        track_ids = []
        for idx in range(result_conf.size):
            conf = float(result_conf[idx])
            if conf >= conf_threshold:
                mask = result["masks"][idx]
                # Scale the result coords to frame coords
                mask[:, 0] *= frame_width / result_width
                mask[:, 1] *= frame_height / result_height
                runner_masks.append(mask)
                confs.append(conf)
                track_ids.append(
                    result["track_ids"][idx] if idx < len(result["track_ids"]) else -1
                )
            else:
                self._logger.info(
                    f"Runner mask prediction ignored due to low confidence: {conf} < {conf_threshold}"
                )
        return runner_masks, confs, track_ids

    async def _get_runner_representative_points(
        self,
        runner_masks: List[np.ndarray],
        bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Optional[Tuple[int, int]]]:
        def get_contour_centers(
            contours: List[np.ndarray],
            bounds: Optional[Tuple[int, int, int, int]] = None,
        ):
            centers = []
            for contour in contours:
                center = contour_center(contour, bounds)
                centers.append(
                    (round(center[0]), round(center[1])) if center is not None else None
                )
            return centers

        runner_points = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(get_contour_centers, runner_masks, bounds),
        )
        return runner_points
