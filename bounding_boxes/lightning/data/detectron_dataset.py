import json
from bounding_boxes.bounding_box_utils import (
    BoundingBox,
    get_csv_string_from_bounding_boxes,
    draw_bounding_boxes,
)
from bounding_boxes.lightning.data.dataloader_wrapper import DataloaderWrapper
from pytorch_lightning import LightningDataModule
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper, MetadataCatalog
from detectron2.config import get_cfg


class Detectron2DataModule(LightningDataModule):
    def __init__(self, cfg, json_files, image_roots, batch_size=1):
        super().__init__()
        self.cfg = cfg
        self.json_files = json_files
        self.image_roots = image_roots
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Register datasets
        dataset_names = ["my_dataset_train", "my_dataset_val", "my_dataset_test"]
        for name, json_file, image_root in zip(
            dataset_names,
            [self.json_files["train"], self.json_files["val"], self.json_files["test"]],
            [
                self.image_roots["train"],
                self.image_roots["val"],
                self.image_roots["test"],
            ],
        ):
            register_coco_instances(name, {}, json_file, image_root)
            num_images = self.count_images(json_file)
            MetadataCatalog.get(name).set(num_images=num_images)

    def count_images(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return len(data["images"])

    def get_mapper(self, is_train=True):
        return DatasetMapper(self.cfg, is_train=is_train)

    def train_dataloader(self):
        self.cfg.DATASETS.TRAIN = ("my_dataset_train",)
        dataloader = build_detection_train_loader(
            self.cfg,
            mapper=self.get_mapper(is_train=True),
            total_batch_size=self.batch_size,
        )
        return dataloader
        # train_dataloader = DataloaderWrapper(dataloader, "my_dataset_train")
        # return train_dataloader

    def val_dataloader(self):
        self.cfg.DATASETS.TEST = ("my_dataset_val",)
        dataloader = build_detection_test_loader(
            self.cfg,
            "my_dataset_val",
            mapper=self.get_mapper(is_train=False),
            batch_size=self.batch_size,
        )
        return dataloader
        # val_dataloader = DataloaderWrapper(dataloader, "my_dataset_val")
        # return val_dataloader

    def test_dataloader(self):
        self.cfg.DATASETS.TEST = ("my_dataset_test",)
        dataloader = build_detection_test_loader(
            self.cfg,
            "my_dataset_test",
            mapper=self.get_mapper(is_train=False),
            batch_size=self.batch_size,
        )
        return dataloader
        # test_dataloader = DataloaderWrapper(dataloader, "my_dataset_test")
        # return test_dataloader


def fetch_and_visualize(dataset_name, idx=0):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    import cv2
    from detectron2.utils.visualizer import Visualizer

    """
    Fetches an item from the dataset by index and visualizes it.

    Args:
    - dataset_name (str): The name of the registered dataset.
    - idx (int): Index of the item to fetch and visualize.
    """
    # Load the dataset and metadata
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # Fetch the specified dataset item
    item = dataset_dicts[idx]

    # Load the image
    img = cv2.imread(item["file_name"])

    bboxes = []
    for ann in item["annotations"]:
        bbox = ann["bbox"]
        bboxes.append(
            BoundingBox(bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2], True)
        )

    bbox_csv = get_csv_string_from_bounding_boxes(bboxes)
    from PIL import Image

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    bbox_img, _, _, _ = draw_bounding_boxes(bbox_csv, pil_img, 1)
    bbox_img.save("coco_example_pil.png")

    # Create a visualizer object and draw the annotations on the image
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
    vis = visualizer.draw_dataset_dict(item)

    # Display the image
    cv2.imwrite("coco_example.png", vis.get_image()[:, :, ::-1])


if __name__ == "__main__":
    cfg = get_cfg()
    config_path = "/home/waynechi/dev/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(config_path)
    # Example setup
    base_json_path = "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/comcrawl/static_comcrawl_no_btn_1_coco_{}.json"
    json_files = {
        "train": base_json_path.format("train"),
        "val": base_json_path.format("val"),
        "test": base_json_path.format("test"),
        "all": "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/comcrawl/static_comcrawl_no_btn_1_coco.json",
    }

    img_path = "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/comcrawl/comcrawl_no_btn_1/images"
    image_roots = {
        "train": img_path,
        "val": img_path,
        "test": img_path,
        "all": img_path,
    }

    data_module = Detectron2DataModule(
        cfg, json_files=json_files, image_roots=image_roots, batch_size=2
    )
    data_module.setup()

    dataset_name = "my_dataset_train"  # Adjust based on how you registered your dataset
    fetch_and_visualize(dataset_name, idx=0)  # Adjust the idx as needed
