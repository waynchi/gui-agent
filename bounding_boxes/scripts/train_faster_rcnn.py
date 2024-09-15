from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from bounding_boxes.lightning.data.detectron_dataset import Detectron2DataModule
import os
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument(
    "--config_path",
    type=str,
    default="/home/waynechi/dev/gui-agent/bounding_boxes/cfg/faster_rcnn.yaml",
)
argparse.add_argument("--max_iter", type=int, default=5000)
argparse.add_argument("--output_dir", type=str, required=True)

args = argparse.parse_args()


config_path = args.config_path

cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.VAL = ("my_dataset_val",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.REFERENCE_WORLD_SIZE = 4
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.SOLVER.MAX_ITER = args.max_iter
cfg.OUTPUT_DIR = args.output_dir

# Skip setting cfg.MODEL.WEIGHTS to not load pre-trained weights

# base_json_path = "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/comcrawl/static_comcrawl_no_btn_1_classes_2_coco_{}.json"
base_json_path = "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/vwa_2/static_vwa_crawl_coco{}.json"
json_files = {
    "train": base_json_path.format("_train"),
    "val": base_json_path.format("_val"),
    "test": base_json_path.format("_test"),
    "all": base_json_path.format(""),
    # "all": "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/comcrawl/static_comcrawl_no_btn_1_classes_2_coco.json",
}

img_path = "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/vwa_2/vwa_crawl/images"
image_roots = {"train": img_path, "val": img_path, "test": img_path, "all": img_path}

data_module = Detectron2DataModule(
    cfg, json_files=json_files, image_roots=image_roots, batch_size=128
)
data_module.setup()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
with open("{}/config.yaml".format(cfg.OUTPUT_DIR), "w") as f:
    f.write(cfg.dump())

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
