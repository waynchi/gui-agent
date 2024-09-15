import os
import json
from bounding_boxes.bounding_box_utils import BoundingBox
from datasets import Dataset, load_dataset
from PIL import Image


def parse_bounding_boxes(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    bounding_boxes = []
    for key, entry in data.items():
        top_left = entry["top_left"]
        bottom_right = entry["bottom_right"]

        # Instantiate BoundingBox
        bbox = BoundingBox(
            top=top_left[1],
            left=top_left[0],
            bottom=bottom_right[1],
            right=bottom_right[0],
            interactable=bool(entry.get("valid", True)),
            class_id=0,
            class_type="UI_ELEMENT",
            text=entry.get("label", ""),
        )
        bounding_boxes.append(bbox)

    return bounding_boxes


def create_coco_json(dataset, base_path):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "UI_ELEMENT"}],
    }
    annotation_id = 0
    image_id = 0
    image_path_to_id = {}

    for i, item in enumerate(dataset):
        image_path = os.path.join(base_path, item["image"])
        box_path = os.path.join(base_path, item["box"])

        # Ensure the image file exists
        if not os.path.exists(image_path):
            image_name = image_path.split("/")[-1]
            image_path = image_path.replace(image_name, image_name.replace("_", ""))

        # Ensure the bounding box file exists
        if not os.path.exists(box_path):
            box_name = box_path.split("/")[-1]
            box_path = box_path.replace(
                box_name, box_name.replace("_", "").replace(".json", "_boxes.json")
            )

        bounding_boxes = parse_bounding_boxes(box_path)
        if image_path not in image_path_to_id:
            image_path_to_id[image_path] = image_id
            image_id += 1
            image = Image.open(image_path)
            image_height = image.size[1]
            image_width = image.size[0]
            # Add image information
            coco_format["images"].append(
                {
                    "id": image_path_to_id[image_path],
                    "file_name": image_path,
                    "height": image_height,
                    "width": image_width,
                }
            )

        # Add annotations for each bounding box
        for bbox in bounding_boxes:
            coco_format["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_path_to_id[image_path],
                    "category_id": 0,
                    "bbox": [
                        bbox.left,
                        bbox.top,
                        bbox.right - bbox.left,
                        bbox.bottom - bbox.top,
                    ],
                    "area": (bbox.right - bbox.left) * (bbox.bottom - bbox.top),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return coco_format


base_path = os.path.expanduser("~/dev/omniact")
data_files = {
    "train": os.path.join(base_path, "train_processed.json"),
    "test": os.path.join(base_path, "test_processed.json"),
    "val": os.path.join(base_path, "val_processed.json"),
}

dataset_list = load_dataset(
    "json", data_files=data_files, split=["train", "test", "val"]
)

dataset = {
    "train": dataset_list[0],
    "test": dataset_list[1],
    "val": dataset_list[2],
}

for split_name, split_data in dataset.items():
    coco_data = create_coco_json(split_data, base_path)
    with open(os.path.join(base_path, f"coco_{split_name}.json"), "w") as f:
        json.dump(coco_data, f, indent=4)
