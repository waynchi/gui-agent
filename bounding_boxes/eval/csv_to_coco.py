import numpy as np
from pycocotools.coco import COCO
import pandas as pd
from detectron2.structures import BoxMode
import os
import json

CLASS_TYPE_TO_ID = {"UI_ELEMENT": 0}


def csv_to_coco(input_path, output_path):
    df = pd.read_csv(input_path)  # Replace with your actual DataFrame loading method

    categories = [
        {"id": class_id, "name": class_type}
        for class_type, class_id in CLASS_TYPE_TO_ID.items()
    ]

    # Initialize COCO dataset structure with the categories filled in
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    # Helper variables
    annotation_id = 0
    image_id = 0

    # Process each row in the DataFrame
    image_id_dict = {}
    for _, row in df.iterrows():
        # Depends on if you want the full path or not
        # file_name = row["image"].split("/")[-1]
        file_name = row["image"]
        image_width = 1280
        image_height = 2048

        # Add image information
        if file_name not in image_id_dict:
            image_id_dict[file_name] = image_id
            coco_dataset["images"].append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": image_width,
                    "height": image_height,
                }
            )
            image_id += 1

        bbox_width = row["right"] - row["left"]
        bbox_height = row["bottom"] - row["top"]

        # Add annotation information
        bbox = [row["left"], row["top"], bbox_width, bbox_height]  # COCO bbox format

        coco_dataset["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id_dict[file_name],
                "category_id": int(row["class_id"]),
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "area": bbox_width * bbox_height,
                "iscrowd": 0,
            }
        )

        annotation_id += 1

    # Save COCO dataset to JSON file
    with open(output_path, "w") as f:
        json.dump(coco_dataset, f, indent=4)


def save_coco(file_name, images, annotations, categories):
    with open(file_name, "w") as f:
        json.dump(
            {"images": images, "annotations": annotations, "categories": categories}, f
        )


def filter_annotations_by_images(annotations, image_ids):
    return [
        annotation for annotation in annotations if annotation["image_id"] in image_ids
    ]


def split_coco_dataset(csv_path, coco_dataset_path):
    df = pd.read_csv(csv_path)
    df["image_filename"] = df["image"].apply(lambda x: os.path.basename(x))

    # Load the original COCO dataset
    coco = COCO(coco_dataset_path)

    # Extract information
    images = coco.dataset["images"]
    annotations = coco.dataset["annotations"]
    categories = coco.dataset["categories"]

    # Shuffle images to ensure randomization
    np.random.seed(777)  # Set seed for reproducibility
    np.random.shuffle(images)

    # Define split ratios for train, val, and test
    train_ratio = 0.8
    val_ratio = 0.1
    # The remainder will be for the test set

    # Calculate split sizes
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    # Split the images
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Get image IDs for filtering annotations
    train_image_ids = set(img["id"] for img in train_images)
    val_image_ids = set(img["id"] for img in val_images)
    test_image_ids = set(img["id"] for img in test_images)

    # Filter annotations for each subset based on image IDs
    train_annotations = filter_annotations_by_images(annotations, train_image_ids)
    val_annotations = filter_annotations_by_images(annotations, val_image_ids)
    test_annotations = filter_annotations_by_images(annotations, test_image_ids)

    # Save the new COCO datasets
    print(
        "Train images: {}, annotations: {}".format(
            len(train_images), len(train_annotations)
        )
    )
    base_path = coco_dataset_path.rsplit(".json", 1)[0]
    save_coco(f"{base_path}_train.json", train_images, train_annotations, categories)
    train_df = df[df["image_filename"].isin([img["file_name"] for img in train_images])]
    train_df.to_csv(f"{base_path.split('coco')[0]}train.csv", index=False)
    print(
        "Val images: {}, annotations: {}".format(len(val_images), len(val_annotations))
    )
    save_coco(f"{base_path}_val.json", val_images, val_annotations, categories)
    val_df = df[df["image_filename"].isin([img["file_name"] for img in val_images])]
    val_df.to_csv(f"{base_path.split('coco')[0]}val.csv", index=False)
    print(
        "Test images: {}, annotations: {}".format(
            len(test_images), len(test_annotations)
        )
    )
    save_coco(f"{base_path}_test.json", test_images, test_annotations, categories)
    test_df = df[df["image_filename"].isin([img["file_name"] for img in test_images])]
    test_df.to_csv(f"{base_path.split('coco')[0]}test.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    csv_to_coco(args.input_path, args.output_path)
    split_coco_dataset(args.input_path, args.output_path)
