import os
import cv2
import glob
from ml_model.yolo import YoloBaseModel


class YoloContours(YoloBaseModel):
    def get_centroids(self, image):
        res = model(image)
        point_list = []
        if res[0].masks:
            for cords in res[0].masks.xy:
                polygon = Polygon(cords)
                point_list.append((polygon.centroid.x, polygon.centroid.y))
        return point_list

    @staticmethod
    def name():
        return "yolo_contours"

    def default_class_map(self):
        return {0: "Runner"}

    def default_yaml(self):
        return "yolov8-seg.yaml"

    @staticmethod
    def create_yolo_labels(submask_dir, yolo_label_dir):
        if not os.path.exists(yolo_label_dir):
            os.mkdir(yolo_label_dir)
        all_submasks = glob.glob(os.path.join(submask_dir, "*"))
        for mask_set_dir in all_submasks:
            img_name = os.path.split(mask_set_dir)[-1]
            all_masks = glob.glob(os.path.join(mask_set_dir, "*"))
            yolo_label = os.path.join(yolo_label_dir, img_name + ".txt")
            with open(yolo_label, "w") as yolo_file:
                for mask_path in all_masks:
                    mask = cv2.imread(mask_path)
                    ret, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
                    mask = mask[:, :, 0]
                    contours = cv2.findContours(
                        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )
                    height, width = mask.shape
                    for contour in contours[0]:
                        # Currently don't support multiple label classifications
                        yolo_file.write("0")
                        for point in contour:
                            yolo_file.write(
                                f" {point[0][0]/width} {point[0][1]/height}"
                            )
                        yolo_file.write("\n")
