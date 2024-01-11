#This is a walkthorugh to train different models and variables for runner detection 

# WARNING If using the ml_model as a package when making changes uninstall the model and run ./install_ml_model.sh 

# Train and find metrics for Mask RCNN (500 dataset)
- Change the size in mask_rcnn.py to (720, 960)

- For augmentation make the train dataloader use AlbumRandAugment(basic_count=2, complex_count=2) 
python ml_model/ml_model/model_util.py --model_type torch_mask_rcnn  --data_folder segmentation_data --weights_name RunnerTorchMaskRCNN500Augmented.pt train

- For unaugmentation make the train dataloader use AlbumRandAugment(basic_count=0, complex_count=0) 
python ml_model/ml_model/model_util.py --model_type torch_mask_rcnn  --data_folder segmentation_data --weights_name RunnerTorchMaskRCNN500.pt train

- Calculate the MAP scores for the different models 
python ml_model/ml_model/model_util.py --model_type torch_mask_rcnn  --data_folder segmentation_data --weights_name RunnerTorchMaskRCNN500Augmented.pt eval
python ml_model/ml_model/model_util.py --model_type torch_mask_rcnn  --data_folder segmentation_data --weights_name RunnerTorchMaskRCNN500.pt eval

# Mask RCNN for 1800 dataset
- Change the size in mask_rcnn.py to (768, 1024)

- Use the option --data_folder runner_data and the option --weights_name RunnerTorchMaskRCNN1800.pt 

- Follow above for training and calculating metrics for both augmented and unaugmented data 


# Train and find metrics Yolo Contours (500 dataset)
- Make sure the size in yolo.py is imgsz=(960, 720)
- python ml_model/ml_model/model_util.py --model_type yolo_contours  --data_folder segmentation_data --weights_name RunnerYoloContour500.pt train 
- Manually move the weights file from ./ml_models/data_store/segmentation_data/ultralytics/runs/segment/train{run idx}/weights/best.pt to ./ml_model/data_store/weights/RunnerYoloContour500.pt
- Find metrics with python ml_model/ml_model/model_util.py --model_type yolo_contours  --data_folder segmentation_data --weights_name RunnerYoloContour500.pt eval 

# Yolo Contours for 1800 dataset 
- Change the size in yolo.py to (720, 960)
- use option --data_folder runner_data and the option --weights_name RunnerYoloContour1800.pt 


