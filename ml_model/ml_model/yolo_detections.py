from ml_model.yolo import YoloBaseModel


class YoloDetections(YoloBaseModel):
    def get_centroids(self, image):
        point_list = []
        res = self.model(image)
        if len(res) <= 0 or not res[0].boxes or len(res[0].boxes.xywh) <= 0:
            return point_list

        for box in res[0].boxes.xywh:
            box_np = box.cpu().numpy().astype(float)
            point_list.append((box_np[0], box_np[1]))

        return point_list

    @staticmethod
    def name():
        return "yolo_detections"

    def default_class_map(self):
        return {0: "Laser"}

    def default_yaml(self):
        return "yolov8n.yaml"
