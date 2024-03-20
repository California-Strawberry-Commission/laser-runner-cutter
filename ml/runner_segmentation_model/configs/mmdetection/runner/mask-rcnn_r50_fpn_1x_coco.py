_base_ = "../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"

# Modify num_classes in model head to match the dataset's annotation
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1))
)
