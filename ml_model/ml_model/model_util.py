import argparse
import os
from ml_model.model_base import ModelBase
from ml_model.model_lookup import all_subclasses, model_lookup
from ml_model.yolo import YoloBaseModel


def setup_parser():
    parser = argparse.ArgumentParser(description="Model training and testing utility")
    # To do, add an argument for if the user wants to run from somewhere other then the ./ml_model/data_store/ folder
    parser.add_argument(
        "--data_store",
        help="Path to data_store like folder structure",
        default="./ml_model/data_store/",
    )
    parser.add_argument(
        "--data_folder", help="Folder for the data set to use", default="runner_data"
    )
    parser.add_argument(
        "--img_folder", help="Folder containing images", default="images"
    )
    parser.add_argument(
        "--seg_mask_folder",
        help="Folder containing full segmentation masks. Segmentation masks are a single mask per image.",
        default="seg_masks",
    )
    parser.add_argument(
        "--mrcnn_mask_folder",
        help="Folder containing mask subdirectories, where each subdirectory contains a one mask image per object instance.",
        default="mrcnn_masks",
    )
    parser.add_argument(
        "--yolo_label_dir",
        default="yolo_labels",
    )

    all_cls_names = [
        model_cls.name() for model_cls in all_subclasses(ModelBase) if model_cls.name()
    ]
    parser.add_argument(
        "--model_type",
        help="Model Type, name of the model to train or evaluate",
        required=True,
        choices=all_cls_names,
    )
    parser.add_argument("--weights_name", help="Name of the weights file")
    parser.add_argument(
        "--weights_folder",
        help="Folder withing the data folder that contains weights file",
        default="weights",
    )
    subparsers = parser.add_subparsers(title="subcommands", dest="command")

    subparsers.add_parser(name="train", help="")
    subparsers.add_parser(
        name="eval",
        help="",
    )
    subparsers.add_parser(name="create_yolo_labels")

    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    project_path = os.path.join(args.data_store, args.data_folder)
    img_dir = os.path.join(project_path, args.img_folder)
    model_cls = model_lookup(args.model_type)
    model = model_cls()
    submask_dir = os.path.join(project_path, args.mrcnn_mask_folder)
    # Yolo labels can be created before any model weights etc. are required
    if args.command == "create_yolo_labels":
        model.create_yolo_labels(
            submask_dir, os.path.join(project_path, args.yolo_label_dir)
        )
        return

    weights_path = os.path.join(args.data_store, args.weights_folder, args.weights_name)
    if os.path.exists(weights_path):
        model.load_weights(weights_path)

    # Could create an data initialization method that other models pass instead if desired.
    if isinstance(model, YoloBaseModel):
        yolo_label_dir = os.path.join(project_path, args.yolo_label_dir)
        model.create_temp_split(project_path, img_dir, yolo_label_dir)

    if args.command == "train":
        model.train_model(img_dir, submask_dir, weights_path)
    elif args.command == "eval":
        map_values = model.get_map_value(img_dir, submask_dir)
        print(map_values)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
