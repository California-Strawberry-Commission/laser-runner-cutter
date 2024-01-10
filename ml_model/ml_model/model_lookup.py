# Note: Currently all the classes need to be imported for the all subclasses call to find them
from ml_model.model_base import ModelBase

# Ran into an error with the semantic segmentation import
# from ml_model.semantic_seg import SemanticSegmentation
from ml_model.mask_rcnn import MaskRCNN
from ml_model.yolo_detections import YoloDetections
from ml_model.yolo_keypoints import YoloKeypoints
from ml_model.yolo_contours import YoloContours


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


def model_lookup(model_name):
    model_types = {
        model_cls.name(): model_cls
        for model_cls in all_subclasses(ModelBase)
        if model_cls.name() is not None
    }
    model_cls = model_types.get(model_name)
    if model_cls is None:
        raise RuntimeError(
            f"No model type {model_name}, choose from {model_types.keys()}"
        )
    return model_cls
