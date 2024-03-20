_base_ = "../point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py"

# Modify num_classes in model head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1),
        point_head=dict(num_classes=1),
    )
)
