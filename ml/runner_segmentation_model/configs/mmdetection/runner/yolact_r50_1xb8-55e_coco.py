_base_ = "../yolact/yolact_r50_1xb8-55e_coco.py"

# Modify num_classes in model head to match the dataset's annotation
model = dict(bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1))
