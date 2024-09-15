import os
import random
import json
from bounding_boxes.bounding_box_utils import BoundingBox
from datasets import Dataset, load_dataset
from pprint import pprint
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


def create_coco_json(dataset, base_path, split, test_set_image_paths):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "UI_ELEMENT"}],
    }
    annotation_id = 0
    image_id = 0
    image_path_to_id = {}
    image_path_count = {}
    image_path_to_task_ids = {}
    box_path_set = set()

    breakpoint()

    for i, item in enumerate(dataset):
        image_path = os.path.join(base_path, item["image"])

        # Ensure the image file exists
        if not os.path.exists(image_path):
            image_name = image_path.split("/")[-1]
            image_path = image_path.replace(image_name, image_name.replace("_", ""))

        if (
            image_path not in image_path_to_id
            and image_path not in test_set_image_paths
        ):
            image_path_to_id[image_path] = image_id
            image_path_count[image_path] = 0
            image_path_to_task_ids[image_path] = []
            image_id += 1

        if image_path in image_path_count:
            image_path_count[image_path] += 1
            image_path_to_task_ids[image_path].append(i)

    # Get images from the dataset such that there are 20 imgs
    image_folder_path_set = set()
    if split == "test":
        # Loop through image_path_to_id
        test_size = 0
        desired_num_apps = 10
        # Loop through image_path_to_id in a random order
        image_paths = list(image_path_to_id.keys())
        print(len(image_paths))
        breakpoint()
        # random.seed(77)
        random.seed(1)  # for a mostly desktop app split
        # random.seed(7)  # for a half and half split
        random.shuffle(image_paths)
        test_set_task_ids = []

        for image_path in image_paths:
            # if len(test_set_image_paths) >= desired_num_apps:
            #     break

            image_folder_path = "/".join(image_path.split("/")[:-1])
            # if image_folder_path in image_folder_path_set:
            #     # This prevents the same application from showing up in the test set twice
            #     continue
            image_folder_path_set.add(image_folder_path)

            test_set_image_paths.append(image_path)
            test_set_task_ids.extend(image_path_to_task_ids[image_path])

            # find the number of tasks associated with the image_path
            test_size += image_path_count[image_path]
            # print("image_path: {}".format(image_path))
            # print("test size: {}".format(test_size))

            # Add image information
            image = Image.open(image_path)
            image_height = image.size[1]
            image_width = image.size[0]
            coco_format["images"].append(
                {
                    "id": image_path_to_id[image_path],
                    "file_name": image_path,
                    "height": image_height,
                    "width": image_width,
                }
            )

        # Add all images in the same app to the test set
        # for image_path in image_paths:
        #     image_folder_path = "/".join(image_path.split("/")[:-1])
        #     if (
        #         image_path in test_set_image_paths
        #         or image_folder_path not in image_folder_path_set
        #     ):
        #         continue

        #     test_set_image_paths.append(image_path)
        #     test_set_task_ids.extend(image_path_to_task_ids[image_path])

        #     # find the number of tasks associated with the image_path
        #     test_size += image_path_count[image_path]
        #     print("image_path: {}".format(image_path))
        #     print("test size: {}".format(test_size))

        #     # Add image information
        #     image = Image.open(image_path)
        #     image_height = image.size[1]
        #     image_width = image.size[0]
        #     coco_format["images"].append(
        #         {
        #             "id": image_path_to_id[image_path],
        #             "file_name": image_path,
        #             "height": image_height,
        #             "width": image_width,
        #         }
        #     )

        test_set_image_paths = sorted(test_set_image_paths)
        print("Number of images in test set: {}".format(len(test_set_image_paths)))
        print("Number of tasks in test set: {}".format(test_size))
        # Print the keys in image_path_to_task_ids
        # print(test_set_image_paths)
        # Count the number of times 'web' appears in test_set_image_paths text
        web_img_count = 0
        web_task_count = 0
        desktop_img_count = 0
        desktop_task_count = 0
        for image_path in test_set_image_paths:
            if "web" in image_path:
                web_img_count += 1
                web_task_count += image_path_count[image_path]
            elif "desktop" in image_path:
                desktop_img_count += 1
                desktop_task_count += image_path_count[image_path]
        print(
            "web: {}, {}, desktop: {}, {}".format(
                web_img_count, web_task_count, desktop_img_count, desktop_task_count
            )
        )

        # Print the values in image_path_to_task_ids in a list
        # print([str(x) for x in sorted(test_set_task_ids)])
        # Output to a file
        # with open(os.path.join(base_path, f"task_ids_diff_app_even.json"), "w") as f:
        with open(os.path.join(base_path, f"task_ids_all.json"), "w") as f:
            json.dump(test_set_task_ids, f, indent=4)

    else:
        for image_path, image_id in image_path_to_id.items():
            image = Image.open(image_path)
            image_height = image.size[1]
            image_width = image.size[0]
            coco_format["images"].append(
                {
                    "id": image_path_to_id[image_path],
                    "file_name": image_path,
                    "height": image_height,
                    "width": image_width,
                }
            )

    for i, item in enumerate(dataset):
        image_path = os.path.join(base_path, item["image"])

        # Ensure the image file exists
        if not os.path.exists(image_path):
            image_name = image_path.split("/")[-1]
            image_path = image_path.replace(image_name, image_name.replace("_", ""))

        if split == "test" and image_path not in test_set_image_paths:
            continue
        if image_path not in image_path_to_id:
            continue

        box_path = os.path.join(base_path, item["box"])

        # Ensure the bounding box file exists
        if not os.path.exists(box_path):
            box_name = box_path.split("/")[-1]
            box_path = box_path.replace(
                box_name, box_name.replace("_", "").replace(".json", "_boxes.json")
            )

        if box_path in box_path_set:
            continue
        box_path_set.add(box_path)

        bounding_boxes = parse_bounding_boxes(box_path)

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

    print(split)
    print("Number of images in coco format: {}".format(len(coco_format["images"])))
    print(
        "Number of annotations in coco format: {}".format(
            len(coco_format["annotations"])
        )
    )

    return coco_format, test_set_image_paths


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

# First create the test set by filtering out images...

test_set_image_paths = []
coco_data, test_set_image_paths = create_coco_json(
    dataset["test"], base_path, "test", test_set_image_paths
)
# with open(os.path.join(base_path, f"coco_diff_app_even_test.json"), "w") as f:
with open(os.path.join(base_path, f"coco_all_test.json"), "w") as f:
    json.dump(coco_data, f, indent=4)

for split_name, split_data in dataset.items():
    if split_name == "test":
        continue
    coco_data, _ = create_coco_json(
        split_data, base_path, split_name, test_set_image_paths
    )
    with open(
        # os.path.join(base_path, f"coco_diff_app_even_{split_name}.json"), "w"
        os.path.join(base_path, f"coco_all_{split_name}.json"),
        "w",
    ) as f:
        json.dump(coco_data, f, indent=4)
