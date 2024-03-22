import os
import argparse
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.apis import DetInferencer
from time import perf_counter

PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_DATA_DIR = os.path.join(
    PROJECT_PATH,
    "data",
    "prepared",
    "runner1800",
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_PATH, "output", "mmdetection")

DATASET_TYPE = "CocoDataset"
DATASET_METAINFO = {
    "classes": ("runner",),
}
CUSTOM_MODELS_DIR = os.path.join(PROJECT_PATH, "configs", "mmdetection", "runner")
MODELS = {
    "mask-rcnn_r50": os.path.join(CUSTOM_MODELS_DIR, "mask-rcnn_r50_fpn_1x_coco.py"),
    "mask-rcnn_r50_albu": os.path.join(
        CUSTOM_MODELS_DIR, "mask-rcnn_r50_fpn_albu_1x_coco.py"
    ),
    "mask-rcnn_r101": os.path.join(CUSTOM_MODELS_DIR, "mask-rcnn_r101_fpn_1x_coco.py"),
    "point-rend_r50": os.path.join(
        CUSTOM_MODELS_DIR, "point-rend_r50-caffe_fpn_ms-1x_coco.py"
    ),
    "rtmdet_tiny": os.path.join(
        CUSTOM_MODELS_DIR, "rtmdet-ins_tiny_8xb32-300e_coco.py"
    ),
    "rtmdet_s": os.path.join(CUSTOM_MODELS_DIR, "rtmdet-ins_s_8xb32-300e_coco.py"),
    "rtmdet_m": os.path.join(CUSTOM_MODELS_DIR, "rtmdet-ins_m_8xb32-300e_coco.py"),
    "rtmdet_l": os.path.join(CUSTOM_MODELS_DIR, "rtmdet-ins_l_8xb32-300e_coco.py"),
    "rtmdet_x": os.path.join(CUSTOM_MODELS_DIR, "rtmdet-ins_x_8xb16-300e_coco.py"),
    "yolact_r50": os.path.join(CUSTOM_MODELS_DIR, "yolact_r50_1xb8-55e_coco.py"),
}


class MMDetection:
    def __init__(
        self,
        model_name,
        data_dir=DEFAULT_DATA_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
    ):
        # Base config
        self.cfg = Config.fromfile(MODELS[model_name])

        # Modify dataset
        if data_dir is not None:
            self.cfg.dataset_type = DATASET_TYPE
            self.cfg.data_root = data_dir

            self.cfg.train_dataloader.dataset.type = DATASET_TYPE
            self.cfg.train_dataloader.dataset.metainfo = DATASET_METAINFO
            self.cfg.train_dataloader.dataset.data_root = data_dir
            self.cfg.train_dataloader.dataset.ann_file = "coco_train.json"
            self.cfg.train_dataloader.dataset.data_prefix = dict(img="images/train")

            self.cfg.val_dataloader.dataset.type = DATASET_TYPE
            self.cfg.val_dataloader.dataset.metainfo = DATASET_METAINFO
            self.cfg.val_dataloader.dataset.data_root = data_dir
            self.cfg.val_dataloader.dataset.ann_file = "coco_val.json"
            self.cfg.val_dataloader.dataset.data_prefix = dict(img="images/val")

            self.cfg.test_dataloader.dataset.type = DATASET_TYPE
            self.cfg.test_dataloader.dataset.metainfo = DATASET_METAINFO
            self.cfg.test_dataloader.dataset.data_root = data_dir
            self.cfg.test_dataloader.dataset.ann_file = "coco_test.json"
            self.cfg.test_dataloader.dataset.data_prefix = dict(img="images/test")

            # Modify evaluator
            self.cfg.val_evaluator.ann_file = os.path.join(data_dir, "coco_val.json")
            self.cfg.test_evaluator.ann_file = os.path.join(data_dir, "coco_test.json")

        # Set up working dir to save files and logs
        if output_dir is not None:
            self.cfg.work_dir = output_dir

    def load_weights(self, weights_file):
        self.cfg.load_from = weights_file

    def train(self, resume=False):
        self.cfg.default_hooks.checkpoint.max_keep_ckpts = 3
        self.cfg.default_hooks.checkpoint.save_best = "coco/segm_mAP"
        self.cfg.resume = resume

        # Enable automatic-mixed-precision (AMP) training
        self.cfg.optim_wrapper.type = "AmpOptimWrapper"
        self.cfg.optim_wrapper.loss_scale = "dynamic"

        # Enable automatically scaling LR
        if (
            "auto_scale_lr" in self.cfg
            and "enable" in self.cfg.auto_scale_lr
            and "base_batch_size" in self.cfg.auto_scale_lr
        ):
            self.cfg.auto_scale_lr.enable = True

        """
        # Models already define num epochs and learning rate, but we can customize it if we want.
        self.cfg.train_cfg.max_epochs = epochs
        self.cfg.param_scheduler = [
            # Linear learning rate warm-up scheduler
            dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
            dict(
                type="MultiStepLR",
                by_epoch=True,
                begin=0,
                end=epochs,
                milestones=[int(epochs * 0.67), int(epochs * 0.92)],
                gamma=0.1,
            ),
        ]
        """

        print(f"Config:\n{self.cfg.pretty_text}")

        # Build the runner from config
        if "runner_type" not in self.cfg:
            # Build the default runner
            runner = Runner.from_cfg(self.cfg)
        else:
            # Build customized runner from the registry if 'runner_type' is set in the cfg
            runner = RUNNERS.build(self.cfg)

        runner.train()

    def eval(self):
        # Build the runner from config
        if "runner_type" not in self.cfg:
            # Build the default runner
            runner = Runner.from_cfg(self.cfg)
        else:
            # Build customized runner from the registry if 'runner_type' is set in the cfg
            runner = RUNNERS.build(self.cfg)

        runner.test()

    def debug(self, image_file, mask_subdir=None):
        # TODO: if mask_subdir is provided, show ground truth and predictions side-by-side
        inferencer = DetInferencer(weights=self.cfg.load_from)

        # Measure inference time
        # Warmup
        for i in range(5):
            inferencer(image_file)
        inference_start = perf_counter()
        inferencer(image_file)
        inference_stop = perf_counter()
        print(f"Inference took {inference_stop - inference_start} seconds.")

        inferencer(image_file, show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MMDetection training, evaluation, and inference"
    )

    subparsers = parser.add_subparsers(title="Available Commands", dest="command")

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--model_name", choices=list(MODELS.keys()), required=True
    )
    train_parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
    )
    train_parser.add_argument("--weights_file")
    train_parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate model performance")
    eval_parser.add_argument("--model_name", choices=list(MODELS.keys()), required=True)
    eval_parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
    )
    eval_parser.add_argument(
        "--weights_file",
        required=True,
    )
    eval_parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    debug_parser = subparsers.add_parser("debug", help="Debug model predictions")
    debug_parser.add_argument(
        "--model_name", choices=list(MODELS.keys()), required=True
    )
    debug_parser.add_argument(
        "--weights_file",
        required=True,
    )
    debug_parser.add_argument("--image_file", required=True)
    debug_parser.add_argument("--mask_subdir")

    args = parser.parse_args()

    data_dir = None if args.command == "debug" else args.data_dir
    output_dir = None if args.command == "debug" else args.output_dir
    model = MMDetection(args.model_name, data_dir=data_dir, output_dir=output_dir)
    weights_file = args.weights_file
    weights_file_exists = weights_file is not None and os.path.exists(weights_file)
    if weights_file_exists:
        model.load_weights(weights_file)

    if args.command == "train":
        model.train(
            resume=weights_file_exists,
        )
    elif args.command == "eval":
        model.eval()
    elif args.command == "debug":
        model.debug(args.image_file, mask_subdir=args.mask_subdir)
    else:
        print("Invalid command.")
