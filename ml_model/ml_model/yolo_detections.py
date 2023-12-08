from ml_model.yolo import YoloBaseModel


class YoloDetections(YoloBaseModel):
    def get_centroids(self, image, score_threshold=0.25):
        point_list = []
        score_list = []
        res = self.model(image)
        if len(res) <= 0 or not res[0].boxes or len(res[0].boxes.xywh) <= 0:
            return score_list, point_list

        for box in res[0].boxes:
            for conf, xywh in zip(
                box.conf.cpu().numpy(), box.xywh.cpu().numpy().astype(float)
            ):
                if conf > score_threshold:
                    x_c, y_c, w, h = xywh
                    point_list.append((x_c, y_c))
                    score_list.append(conf)
        return score_list, point_list

    @staticmethod
    def name():
        return "yolo_detections"

    def default_class_map(self):
        return {0: "Laser"}

    def default_yaml(self):
        return "yolov8n.yaml"
