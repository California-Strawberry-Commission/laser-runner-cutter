from ml_model.yolo import YoloBaseModel


class YoloKeypoints(YoloBaseModel):
    def get_centroids(self, image):
        raise NotImplementedError

    @staticmethod
    def name():
        return "yolo_keypoints"

    def default_class_map(self):
        return {0: "Laser"}

    def default_yaml(self):
        return "yolov8-pose.yaml"
