from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from bounding_boxes.lightning.data.detectron_dataset import Detectron2DataModule
import os
import argparse
from detectron2.engine import DefaultPredictor


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    default="/home/waynechi/dev/gui-agent/bounding_boxes/cfg/faster_rcnn.yaml",
)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--img_path", type=str, required=True)

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

from detectron2.utils.visualizer import ColorMode

from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
from detectron2.utils.visualizer import Visualizer

train_dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
val_dataset = DatasetCatalog.get(cfg.DATASETS.VAL[0])
test_dataset = DatasetCatalog.get(cfg.DATASETS.TEST[0])

train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
val_metadata = MetadataCatalog.get(cfg.DATASETS.VAL[0])
test_metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

# Fetch the specified dataset item
idx = 10
item = test_dataset[idx]
im = cv2.imread(item["file_name"])
outputs = predictor(im)

# TODO Remove this
# outputs["instances"].pred_classes[:] = 0

v = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=1)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("model_pred_example.png", out.get_image()[:, :, ::-1])

# Visualizer for the ground truth
v_gt = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=1)
out_gt = v_gt.draw_dataset_dict(
    item
)  # This function uses the 'annotations' key by default
ground_truth_image_path = "ground_truth_example.png"
cv2.imwrite(ground_truth_image_path, out_gt.get_image()[:, :, ::-1])

# model = predictor.model

# print(predictor.model)
#
# # Use torchinfo to print the summary. Adjust the input_size and depth as needed
# from torchinfo import summary
#
# summary(model)
