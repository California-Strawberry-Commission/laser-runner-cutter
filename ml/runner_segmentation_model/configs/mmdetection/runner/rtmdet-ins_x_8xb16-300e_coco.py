_base_ = "../rtmdet/rtmdet-ins_x_8xb16-300e_coco.py"

# Modify num_classes in model head to match the dataset's annotation
model = dict(bbox_head=dict(num_classes=1))
