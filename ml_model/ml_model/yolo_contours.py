import os
import cv2
import glob
from ml_model.yolo import YoloBaseModel
from shapely import Polygon, Point
from shapely.ops import nearest_points


class YoloContours(YoloBaseModel):
    def get_centroids(self, image):
        res = self.model(image)
        point_list = []
        score_list = []
        if res[0].masks:
            for score, cords in zip(res[0].probs, res[0].masks.xy):
                polygon = Polygon(cords)
                closest_polygon_point, closest_point = nearest_points(
                    polygon, polygon.centroid
                )
                point_list.append((closest_polygon_point.x, closest_polygon_point.y))
                score_list.append(score)

        return score_list, point_list

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
