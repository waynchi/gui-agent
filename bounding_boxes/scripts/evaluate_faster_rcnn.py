from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from bounding_boxes.lightning.data.detectron_dataset import Detectron2DataModule
import os
import argparse
from detectron2.engine import DefaultPredictor


from detectron2.utils.visualizer import ColorMode

from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
import torch
from detectron2.structures import Boxes, Instances


from detectron2.data import build_detection_test_loader
import numpy as np

from detectron2.evaluation import DatasetEvaluator
import numpy as np


class CenterPointEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, output_dir=None):
        super().__init__(dataset_name, cfg, False, output_dir)
        self.center_diffs = []

    def reset(self):
        super().reset()
        self.center_diffs = []

    def process(self, inputs, outputs):
        coco_api = self._coco_api
        for input, output in zip(inputs, outputs):
            ann_ids = coco_api.getAnnIds(imgIds=input["image_id"])
            anno = coco_api.loadAnns(ann_ids)
            gt_boxes = [
                BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                for obj in anno
                if obj["iscrowd"] == 0
            ]
            gt_boxes = torch.as_tensor(gt_boxes).reshape(
                -1, 4
            )  # guard against no boxes
            gt_boxes = Boxes(gt_boxes)
            # Assuming 'instances' is the predicted outputs
            pred_boxes = output["instances"].pred_boxes.tensor.cpu().numpy()
            # gt_boxes = input["instances"].gt_boxes.tensor.cpu().numpy()
            breakpoint()

            # Calculate centers
            pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2

            # Compute center differences
            diff = np.linalg.norm(pred_centers - gt_centers, axis=1)
            self.center_diffs.extend(diff)

    def evaluate(self):
        # Compute the mean center point error
        mean_error = np.mean(self.center_diffs)
        return {"mean_center_error": mean_error}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    default="/home/waynechi/dev/gui-agent/bounding_boxes/cfg/faster_rcnn.yaml",
)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--img_path", type=str, required=True)


# Just for ease
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--base_lr", type=float, default=0.00025)
parser.add_argument("--max_iters", type=int, default=100)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--freeze_backbone", action="store_true")

args = parser.parse_args()

# Initialize W&B logger
# wandb_logger = WandbLogger(name="FasterRCNN_Training", project="ObjectDetection")

config_path = args.config_path
cfg = get_cfg()
cfg.merge_from_file(config_path)
# cfg.SOLVER.MAX_ITER = args.max_epochs * args.image_batches_per_epoch
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.VAL = ("my_dataset_val",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.REFERENCE_WORLD_SIZE = 4
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.OUTPUT_DIR = args.output_dir
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = f"{args.output_dir}/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold

predictor = DefaultPredictor(cfg)

# Example setup
# dataset_path = '/home/waynechi/dev/gui-agent/bounding_boxes/datasets/vwa_2/static_comcrawl_no_btn_1_coco_{}.json'
dataset_path = args.dataset_path
json_files = {
    "train": dataset_path.format("_train"),
    "val": dataset_path.format("_val"),
    "test": dataset_path.format("_test"),
    # "all": dataset_path.format(""),
}

# img_path = "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/comcrawl/comcrawl_no_btn_1/images"
img_path = args.img_path
image_roots = {
    "train": img_path,
    "val": img_path,
    "test": img_path,
    # "all": img_path,
}

data_module = Detectron2DataModule(
    cfg, json_files=json_files, image_roots=image_roots, batch_size=128
)
data_module.setup()

train_dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
val_dataset = DatasetCatalog.get(cfg.DATASETS.VAL[0])
test_dataset = DatasetCatalog.get(cfg.DATASETS.TEST[0])

train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
val_metadata = MetadataCatalog.get(cfg.DATASETS.VAL[0])
test_metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

os.makedirs("eval_output", exist_ok=True)

for split in ["val", "test"]:
    evaluator = COCOEvaluator(
        f"my_dataset_{split}", cfg, False, output_dir="./eval_output"
    )
    evaluator = CenterPointEvaluator(
        f"my_dataset_{split}", cfg, output_dir="./eval_output"
    )
    data_loader = build_detection_test_loader(cfg, f"my_dataset_{split}")

    # Run evaluation
    print(f"Starting evaluation on {split} dataset...")
    eval_results = inference_on_dataset(predictor.model, data_loader, evaluator)

    # This will print the AP metrics at different IoU thresholds
    print(eval_results)
