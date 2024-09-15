"""
Evaluate if the bounding boxes in a particular environment are correct
"""

from PIL import Image, ImageDraw, ImageFont
import easyocr
import shutil
import time
import torch
import sys
import os
import json
import argparse
from definitions import PROJECT_DIR

sys.path.append(os.path.join(PROJECT_DIR, "../webui/models/screenrecognition"))

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import glob
import numpy as np
from ui_models import UIElementDetector
import pytorch_lightning
import pandas as pd
from bounding_boxes.bounding_box_utils import (
    compute_bounding_boxes_webui,
    BoundingBox,
    get_csv_string_from_bounding_boxes,
    draw_bounding_boxes,
    calculate_iou,
)

from bounding_boxes.scripts.bbox_predictor import BoundingBoxPredictor

from bounding_boxes.eval.dataset_generation import (
    MINIWOB_DATASET_PATH,
    WEBARENA_DATASET_PATH,
    COMCRAWL_DATASET_PATH,
    VWA_DATASET_PATH,
    VWA_HTML_DATASET_PATH,
)
from mean_average_precision import MetricBuilder
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from enum import Enum

from bounding_boxes.sam_wrapper import SegmentAnythingWrapper
from bounding_boxes.eval.nlp_metrics import (
    calculate_bleu_from_text,
    calculate_wer_from_text,
)


class BBOX_MODEL_TYPE(str, Enum):
    WEBUI = "webui"
    SAM = "sam"
    SAM_GT_POINTS = "sam_gt_points"  # Default is huge
    SAM_GT_POINTS_VIT_BASE = "sam_gt_points_vit_base"
    SAM_GT_POINTS_VIT_LARGE = "sam_gt_points_vit_large"
    SAM_GT_POINTS_MULTIMASK = "sam_gt_points_multimask"
    SAM_EXTRA_BACKGROUND = "sam_gt_points_extra_background"
    FASTER_RCNN = "faster_rcnn"


def compute_iou(pred, gt):
    """Calculates IoU (Jaccard index) of two sets of bboxes:
        IOU = pred ∩ gt / (area(pred) + area(gt) - pred ∩ gt)

    Parameters:
        Coordinates of bboxes are supposed to be in the following form: [x1, y1, x2, y2]
        pred (np.array): predicted bboxes
        gt (np.array): ground truth bboxes

    Return value:
        iou (np.array): intersection over union
    """

    def get_box_area(box):
        return (box[:, 2] - box[:, 0] + 1.0) * (box[:, 3] - box[:, 1] + 1.0)

    _gt = np.tile(gt, (pred.shape[0], 1))
    _pred = np.repeat(pred, gt.shape[0], axis=0)

    ixmin = np.maximum(_gt[:, 0], _pred[:, 0])
    iymin = np.maximum(_gt[:, 1], _pred[:, 1])
    ixmax = np.minimum(_gt[:, 2], _pred[:, 2])
    iymax = np.minimum(_gt[:, 3], _pred[:, 3])

    width = np.maximum(ixmax - ixmin + 1.0, 0)
    height = np.maximum(iymax - iymin + 1.0, 0)

    intersection_area = width * height
    union_area = get_box_area(_gt) + get_box_area(_pred) - intersection_area
    iou = (intersection_area / union_area).reshape(pred.shape[0], gt.shape[0])
    return iou


def eval_group(ground_truth_df, pred_df, metric_fn):
    column_names = ["left", "top", "right", "bottom", "class_id"]
    ground_truth = ground_truth_df[column_names].to_numpy()
    ground_truth = np.append(ground_truth, np.zeros((ground_truth.shape[0], 2)), axis=1)
    ground_truth = ground_truth.astype(int)

    column_names.append("confidence")
    preds = pred_df[column_names].to_numpy()

    # Assert that every second element is less than the fourth element
    if np.any(preds[:, 0] > preds[:, 2]):
        breakpoint()
    assert np.all(preds[:, 0] <= preds[:, 2])
    if np.any(preds[:, 1] > preds[:, 3]):
        breakpoint()
    assert np.all(preds[:, 1] <= preds[:, 3])
    if np.any(ground_truth[:, 0] > ground_truth[:, 2]):
        breakpoint()
    assert np.all(ground_truth[:, 0] <= ground_truth[:, 2])
    if np.any(ground_truth[:, 1] > ground_truth[:, 3]):
        breakpoint()
    assert np.all(ground_truth[:, 1] <= ground_truth[:, 3])

    metric_fn.add(preds, ground_truth)

    # evaluate_mean_average_precision(miniwob_ground_truth, miniwob_preds)


def get_webui_preds(webui_preds_path, image_folder, merge_iou_thresholds):
    assert os.path.isdir("/".join(webui_preds_path.split("/")[:-1]))
    print("Generating predictions for webui for image folder: {}".format(image_folder))
    pred_dfs = []
    model = UIElementDetector.load_from_checkpoint(
        "checkpoints/screenrecognition-web350k.ckpt"
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")
    model.to(device)
    model.eval()

    # Make sure to generate the dataset first
    # Loop through files in the folder
    i = 0
    for filename in glob.glob(os.path.join(image_folder, "*.png")):
        if i % 100 == 0:
            print("processed {} images".format(i))
        i += 1

        bounding_boxes_dict = compute_bounding_boxes_webui(
            model, filename, merge_iou_thresholds=merge_iou_thresholds
        )
        for merge_iou_threshold in merge_iou_thresholds:
            bounding_boxes = bounding_boxes_dict[merge_iou_threshold]
            pred_df = pd.DataFrame(
                [bounding_box.to_dict() for bounding_box in bounding_boxes]
            )
            pred_df["environment"] = filename.split("/")[-1].split(".")[0]
            pred_df["box_index"] = pred_df.index
            pred_df["image"] = filename
            pred_df["merge_iou_threshold"] = merge_iou_threshold
            pred_dfs.append(pred_df)

    webui_pred_df = pd.concat(pred_dfs)
    webui_pred_df.to_csv(webui_preds_path, index=False)

    return webui_pred_df


def get_points_and_labels(df, use_extra_foreground=False, use_extra_background=False):
    values = df[["center_x", "center_y", "top", "left", "bottom", "right"]].to_numpy()

    if use_extra_foreground:
        raise NotImplementedError
    elif use_extra_background:
        points = np.empty((values.shape[0], 5, 2))

        # Fill the new matrix with the appropriate values
        points[:, 0, :] = values[:, :2]  # center_x, center_y
        points[:, 1, :] = values[:, [2, 3]]  # top, left
        points[:, 2, :] = values[:, [2, 5]]  # top, right
        points[:, 3, :] = values[:, [4, 3]]  # bottom, left
        points[:, 4, :] = values[:, [4, 5]]  # bottom, right
        points = torch.from_numpy(points)
        labels = torch.ones((points.shape[0], 5), dtype=torch.long)
        labels[:, 2:5] = 0
    else:
        points = df[["center_x", "center_y"]].to_numpy()
        points = torch.from_numpy(points)
        points = points.unsqueeze(1)
        labels = torch.ones(points.shape[0], dtype=torch.long)
        labels = labels.unsqueeze(1)

    return points, labels


def get_sam_preds(
    preds_path,
    image_folder,
    gt_df,
    use_gt_points=False,
    use_extra_foreground=False,
    use_extra_background=False,
    use_multimask=False,
):
    assert os.path.isdir("/".join(preds_path.split("/")[:-1]))
    print("Generating predictions for sam for image folder: {}".format(image_folder))
    pred_dfs = []

    # checkpoint = "sam_vit_h_4b8939.pth"
    # model_type = "vit_h"
    # checkpoint = "sam_vit_b_01ec64.pth"
    # model_type = "vit_b"
    checkpoint = "sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")

    sam_wrapper = SegmentAnythingWrapper(checkpoint, model_type, device)

    # Make sure to generate the dataset first
    # Loop through files in the folder
    i = 0
    for filename in glob.glob(os.path.join(image_folder, "*.png")):
        if i % 10 == 0:
            print("processed {} images".format(i))
        i += 1

        cv2_image = cv2.imread(filename)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        if use_gt_points:
            # Get the ground truth points
            image_df = gt_df[gt_df["image"] == filename]

            points, labels = get_points_and_labels(
                image_df,
                use_extra_foreground=use_extra_foreground,
                use_extra_background=use_extra_background,
            )

            bounding_boxes = sam_wrapper.get_bounding_boxes(
                cv2_image, points, labels, use_multimask=use_multimask
            )
        else:
            bounding_boxes = sam_wrapper.get_all_bounding_boxes(cv2_image)

        pred_df = pd.DataFrame(
            [bounding_box.to_dict() for bounding_box in bounding_boxes]
        )
        pred_df["environment"] = filename.split("/")[-1].split(".")[0]
        pred_df["box_index"] = pred_df.index
        pred_df["image"] = filename
        pred_df["merge_iou_threshold"] = 1
        pred_dfs.append(pred_df)

    pred_df = pd.concat(pred_dfs)
    pred_df.to_csv(preds_path, index=False)

    return pred_df


def get_faster_rcnn_preds(
    preds_path, image_folder, predictor, ground_truth_df, confidence_threshold=0.0
):
    assert os.path.isdir("/".join(preds_path.split("/")[:-1]))
    print(
        "Generating predictions for faster rcnn for image folder: {}".format(
            image_folder
        )
    )
    pred_dfs = []

    print("CUDA is available: {}".format(torch.cuda.is_available()))

    # Make sure to generate the dataset first
    # Loop through files in the folder
    i = 0

    all_files = glob.glob(os.path.join(image_folder, "*.png"))
    unique_images = ground_truth_df["image"].unique()
    filtered_files = [filename for filename in all_files if filename in unique_images]

    print("length of all files: {}".format(len(all_files)))
    print("length of filtered files: {}".format(len(filtered_files)))

    for filename in filtered_files:
        if i % 10 == 0:
            print("processed {} images".format(i))
        i += 1

        cv2_image = cv2.imread(filename)

        bounding_boxes = predictor.predict(cv2_image, confidence_threshold)

        pred_df = pd.DataFrame(
            [bounding_box.to_dict() for bounding_box in bounding_boxes]
        )
        pred_df["environment"] = filename.split("/")[-1].split(".")[0]
        pred_df["box_index"] = pred_df.index
        pred_df["image"] = filename
        pred_df["merge_iou_threshold"] = 1
        pred_dfs.append(pred_df)

    pred_df = pd.concat(pred_dfs)
    pred_df.to_csv(preds_path, index=False)

    return pred_df


def save_example_image(
    pred_df,
    ground_truth_df,
    image_path,
    output_folder,
    confidence_threshold,
    use_gt_points=False,
    use_foreground_points=False,
    use_background_points=False,
):
    # Get an example of a prediction and ground truth
    pred_df_example = pred_df[pred_df["image"] == image_path]
    ground_truth_df_example = ground_truth_df[ground_truth_df["image"] == image_path]

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Convert bounding box df to
    pred_bboxes = []
    for i, row in pred_df_example.iterrows():
        if row["confidence"] < confidence_threshold:
            continue
        pred_bboxes.append(
            BoundingBox(
                row["top"],
                row["left"],
                row["bottom"],
                row["right"],
                interactable=True,
                class_type=row["class_type"],
                class_id=row["class_id"],
                text=row["text"],
            )
        )

    ground_truth_bboxes = []
    for i, row in ground_truth_df_example.iterrows():
        ground_truth_bboxes.append(
            BoundingBox(
                row["top"],
                row["left"],
                row["bottom"],
                row["right"],
                interactable=True,
                # class_type=row["class_type"],
                # class_id=row["class_id"],
                # text=row["text"],
            )
        )

    # Copy image
    shutil.copyfile(image_path, os.path.join(output_folder, "screenshot.png"))

    pred_csv_string = get_csv_string_from_bounding_boxes(pred_bboxes)
    ground_truth_csv_string = get_csv_string_from_bounding_boxes(ground_truth_bboxes)

    pred_image = Image.open(image_path)

    (
        pred_img,
        _,
        pred_content_str,
        _,
    ) = draw_bounding_boxes(pred_csv_string, pred_image, 1)

    if use_gt_points:
        draw = ImageDraw.Draw(pred_img)
        # The size of the point to draw (as a square around the center point)
        point_size = 3

        points, labels = get_points_and_labels(
            ground_truth_df_example,
            use_extra_foreground=use_foreground_points,
            use_extra_background=use_background_points,
        )

        i = 0
        # Iterate over each point and its label
        for point, label in zip(points, labels.numpy()):
            # Extract center_x and center_y
            center_x, center_y = point[0]

            # Determine the color based on the label
            # Assuming label == 1 for foreground points, and label == 0 for background points
            color = "red" if label[0] == 1 else "blue"

            # Draw a small rectangle around the point to make it visible
            # Adjust the point size as needed
            draw.rectangle(
                [
                    center_x - point_size,
                    center_y - point_size,
                    center_x + point_size,
                    center_y + point_size,
                ],
                fill=color,
            )

            # Position for the number is slightly to the right and above the point
            number_position = (center_x + point_size + 2, center_y - point_size)

            # Draw the number
            font = ImageFont.load_default()
            # draw.text(number_position, str(i), fill=color, font=font)
            i += 1

    pred_img.save(os.path.join(output_folder, "pred.png"))

    ground_truth_image = Image.open(image_path)
    (
        ground_truth_img,
        _,
        gt_content_str,
        _,
    ) = draw_bounding_boxes(ground_truth_csv_string, ground_truth_image, 1)
    ground_truth_img.save(os.path.join(output_folder, "ground_truth.png"))

    print(
        "BLEU Score: {}".format(
            calculate_bleu_from_text([gt_content_str], pred_content_str)
        )
    )
    print(
        "WER Score: {}".format(
            calculate_wer_from_text(gt_content_str, pred_content_str)
        )
    )
    with open(os.path.join(output_folder, "info.json"), "w") as f:
        info = {
            "gt_content": gt_content_str,
            "pred_content": pred_content_str,
        }
        json.dump(info, f, indent=4)


def main():

    bbox_model_type = BBOX_MODEL_TYPE.FASTER_RCNN

    parser = argparse.ArgumentParser()
    parser.add_argument("--additional_params", type=str, default="")
    parser.add_argument(
        "--dataset",
        type=str,
        default="comcrawl_no_btn_1",
        help="[comcrawl_no_btn_1, vwa_no_btn_1]",
    )
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--ground_truth_path", type=str, required=True)

    args = parser.parse_args()
    additional_params = args.additional_params
    # additional_params = "_val_20k_bs_16"

    if "train" in additional_params:
        subset = "train"
    elif "val" in additional_params:
        subset = "val"
    elif "test" in additional_params:
        subset = "test"
    else:
        subset = "dataset"

    if args.dataset == "comcrawl_no_btn_1":
        dataset_path = COMCRAWL_DATASET_PATH
    elif args.dataset == "vwa_no_btn_1" or args.dataset == "vwa_crawl":
        dataset_path = VWA_DATASET_PATH
    elif args.dataset == "vwa_crawl_html":
        dataset_path = VWA_HTML_DATASET_PATH
        args.dataset = "vwa_crawl"

    image_folder = os.path.join(dataset_path, args.dataset, "images")
    preds_path = os.path.join(
        PROJECT_DIR,
        "bounding_boxes/preds/{}/{}{}_preds.csv".format(
            args.dataset, bbox_model_type, additional_params
        ),
    )
    ground_truth_df = pd.read_csv(args.ground_truth_path)
    eval_filename = os.path.join(
        dataset_path,
        "eval_results",
        "{}{}_{}_eval_results.csv".format(
            bbox_model_type, additional_params, args.dataset
        ),
    )

    print("saving predictions to: {}".format(preds_path))
    print("saving eval to: {}".format(eval_filename))

    if bbox_model_type == "webui":
        merge_iou_thresholds = np.round(np.arange(0.3, 0.71, 0.1), 1)
    elif "sam" in bbox_model_type or "faster_rcnn" in bbox_model_type:
        merge_iou_thresholds = np.round(np.arange(1, 1.01, 0.1), 1)

    use_multimask = "multimask" in bbox_model_type
    use_gt_points = "gt_points" in bbox_model_type
    use_extra_background = "extra_background" in bbox_model_type
    use_extra_foreground = "extra_foreground" in bbox_model_type

    # if os.path.isfile(preds_path):
    if False:
        pred_df = pd.read_csv(preds_path)
    else:
        if bbox_model_type == "webui":
            pred_df = get_webui_preds(preds_path, image_folder, merge_iou_thresholds)
        elif "sam" in bbox_model_type:
            pred_df = get_sam_preds(
                preds_path,
                image_folder,
                ground_truth_df,
                use_gt_points=use_gt_points,
                use_extra_background=use_extra_background,
                use_extra_foreground=use_extra_foreground,
                use_multimask=use_multimask,
            )
        elif "faster_rcnn" in bbox_model_type:
            model_path = args.model_path
            # This has to exactly match training
            config_path = "/home/waynechi/dev/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            bbox_predictor = BoundingBoxPredictor(model_path, config_path)

            pred_df = get_faster_rcnn_preds(
                preds_path,
                image_folder,
                bbox_predictor,
                ground_truth_df,
                confidence_threshold=0.0,
            )

    confidence_thresholds = np.round(np.arange(0.3, 0.31, 0.2), 1)

    # Create an empty DataFrame to store the results
    results_df_list = []

    image_path = ground_truth_df["image"].iloc[77]
    save_example_image(
        pred_df,
        ground_truth_df,
        image_path,
        os.path.join(dataset_path, "example_images", bbox_model_type),
        confidence_threshold=0.3,
        use_gt_points=use_gt_points,
        use_background_points=use_extra_background,
        use_foreground_points=use_extra_foreground,
    )

    for merge_iou_threshold in merge_iou_thresholds:
        for confidence_threshold in confidence_thresholds:

            print(
                "merge_iou_threshold: {} | confidence threshold: {}".format(
                    merge_iou_threshold, confidence_threshold
                )
            )
            pred_df_filtered = pred_df[
                pred_df["merge_iou_threshold"] == merge_iou_threshold
            ]
            pred_df_filtered = pred_df_filtered[
                pred_df_filtered["confidence"] > confidence_threshold
            ]

            unique_images = ground_truth_df["image"].unique()
            metric_fn = MetricBuilder.build_evaluation_metric(
                "map_2d", async_mode=True, num_classes=1
            )

            for unique_image in unique_images:
                eval_group(
                    ground_truth_df[ground_truth_df["image"] == unique_image],
                    pred_df_filtered[pred_df_filtered["image"] == unique_image],
                    metric_fn,
                )

            print(
                "finished merging for ioU: {} and confidence threshold: {}",
                merge_iou_threshold,
                confidence_threshold,
            )

            start_time = time.time()
            print("voc pascal map all points time start: {}".format(start_time))
            result = metric_fn.value(iou_thresholds=0.5)
            pascal_voc_map = result["mAP"]
            pascal_voc_max_recall = max(result[0.5][0]["recall"])
            pascal_voc_max_precision = max(result[0.5][0]["precision"])
            print(
                "pascal voc mAP: {} | recall: {} | precision: {}".format(
                    pascal_voc_map, pascal_voc_max_recall, pascal_voc_max_precision
                )
            )
            # voc_pascal_recall = max(result[0.5][0]['recall'])
            end_time = time.time()
            print("voc pascal map all points time end: {}".format(end_time))
            duration = end_time - start_time
            print("voc pascal map all points time: {}".format(duration))

            start_time = time.time()
            print("coco map time start: {}".format(start_time))
            result = metric_fn.value(
                iou_thresholds=np.arange(0.5, 1.0, 0.05),
                recall_thresholds=np.arange(0.0, 1.01, 0.01),
                mpolicy="soft",
            )
            coco_map = result["mAP"]
            coco_max_recall = max(result[0.5][0]["recall"])
            coco_max_precision = max(result[0.5][0]["precision"])
            print(
                "coco mAP: {} | recall: {} | precision: {}".format(
                    coco_map, coco_max_recall, coco_max_precision
                )
            )
            end_time = time.time()
            print("coco map time end: {}".format(end_time))
            duration = end_time - start_time
            print("coco map time: {}".format(duration))

            # Append the results to the DataFrame
            results_df_list.append(
                pd.DataFrame(
                    {
                        "merge_iou_threshold": merge_iou_threshold,
                        "confidence_threshold": confidence_threshold,
                        "coco_mAP": coco_map,
                        "coco_max_recall": coco_max_recall,
                        # "pascal_voc_mAP": voc_pascal_map,
                        "pascal_voc_mAP_all_points": pascal_voc_map,
                        "pascal_voc_max_recall": pascal_voc_max_recall,
                    },
                    index=[0],
                    dtype=np.float32,
                )
            )

    results_df = pd.concat(results_df_list)
    results_df.to_csv(eval_filename, index=False)

    pivot_coco_map = results_df.pivot(
        index="merge_iou_threshold", columns="confidence_threshold", values="coco_mAP"
    )
    # pivot_voc_pascal_map = results_df.pivot(
    #     index="merge_iou_threshold",
    #     columns="confidence_threshold",
    #     values="pascal_voc_mAP",
    # )
    pivot_voc_pascal_map_all_points = results_df.pivot(
        index="merge_iou_threshold",
        columns="confidence_threshold",
        values="pascal_voc_mAP_all_points",
    )

    # Plotting the heatmaps
    plt.figure(figsize=(18, 6))

    # COCO mAP
    plt.subplot(1, 3, 1)
    sns.heatmap(pivot_coco_map, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("COCO mAP")

    # PASCAL VOC mAP
    # plt.subplot(1, 3, 2)
    # sns.heatmap(pivot_voc_pascal_map, annot=True, cmap="YlGnBu", fmt=".2f")
    # plt.title("PASCAL VOC mAP")

    # PASCAL VOC mAP all points
    plt.subplot(1, 3, 3)
    sns.heatmap(pivot_voc_pascal_map_all_points, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("PASCAL VOC mAP all points")

    plt.tight_layout()
    plt.show()
    plt.savefig(
        os.path.join(
            dataset_path,
            "{}_{}_eval_results.png".format(bbox_model_type, args.dataset),
        )
    )


if __name__ == "__main__":
    main()
