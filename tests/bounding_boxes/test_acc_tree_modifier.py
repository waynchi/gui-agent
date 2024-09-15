import os
from PIL import Image
import pandas as pd
from bounding_boxes.scripts.acc_tree_modifier import (
    AccessibilityTreeModifier,
)
from bounding_boxes.bounding_box_utils import draw_bounding_boxes

data_dir = "/home/waynechi/dev/gui-agent/tests/bounding_boxes/data/shopping"


# def test_modify_with_gt():
#     # Load the ground truth and prediction DataFrames
#     gt_df = pd.read_csv(os.path.join(data_dir, "gt_som.csv"))
#     pred_df = pd.read_csv(os.path.join(data_dir, "pred_som.csv"))
#
#     # Initialize an instance of AccessibilityTreeModifier with default settings
#     modifier = AccessibilityTreeModifier(
#         use_bboxes="gt",
#         use_tags="gt",
#         use_interact_element_text="gt",
#         use_static_text="gt",
#         use_ordering="default",
#     )
#
#     # Call the modify function
#     modified_df = modifier.modify(gt_df, pred_df)
#     # modified_df.to_csv("modified_df.csv", index=False)
#
#     # Check if the modified DataFrame is equal to the ground truth DataFrame
#     pd.testing.assert_frame_equal(modified_df, gt_df, check_dtype=False)
#
#
# def test_modify_with_pred():
#     # Load the ground truth and prediction DataFrames
#     gt_df = pd.read_csv(os.path.join(data_dir, "gt_som.csv"))
#     pred_df = pd.read_csv(os.path.join(data_dir, "pred_som.csv"))
#
#     # Initialize an instance of AccessibilityTreeModifier with default settings
#     modifier = AccessibilityTreeModifier(
#         use_bboxes="pred",
#         use_tags="pred",
#         use_interact_element_text="pred",
#         use_static_text="pred",
#         use_ordering="default",
#     )
#
#     # Call the modify function
#     modified_df = modifier.modify(gt_df, pred_df)
#     modified_df.to_csv("modified_df.csv", index=False)
#
#     # Check if the modified DataFrame is equal to the ground truth DataFrame
#     pd.testing.assert_frame_equal(modified_df, pred_df, check_dtype=False)
#
#
# def test_modify_with_origin():
#     # Load the ground truth and prediction DataFrames
#     gt_df = pd.read_csv(os.path.join(data_dir, "gt_som.csv"))
#     pred_df = pd.read_csv(os.path.join(data_dir, "pred_som.csv"))
#     if "Unnamed: 0" in gt_df.columns:
#         gt_df.drop(columns=["Unnamed: 0"], inplace=True)
#     if "Unnamed: 0" in pred_df.columns:
#         pred_df.drop(columns=["Unnamed: 0"], inplace=True)
#
#     # Initialize an instance of AccessibilityTreeModifier with default settings
#     modifier = AccessibilityTreeModifier(
#         use_bboxes="pred",
#         use_tags="pred",
#         use_interact_element_text="pred",
#         use_static_text="pred",
#         use_ordering="origin",
#     )
#
#     # Call the modify function
#     modified_df = modifier.modify(gt_df, pred_df)
#     modified_df.to_csv("modified_df.csv", index=False)
#
#     # Check if the modified DataFrame is equal to the ground truth DataFrame
#     pd.testing.assert_frame_equal(modified_df, pred_df, check_dtype=False)


def test_modify_with_pred():
    # Load the ground truth and prediction DataFrames
    gt_df = pd.read_csv(os.path.join(data_dir, "gt_som.csv"))
    pred_df = pd.read_csv(os.path.join(data_dir, "pred_som.csv"))
    if "Unnamed: 0" in gt_df.columns:
        gt_df.drop(columns=["Unnamed: 0"], inplace=True)
    if "Unnamed: 0" in pred_df.columns:
        pred_df.drop(columns=["Unnamed: 0"], inplace=True)

    screenshot_img = Image.open(os.path.join(data_dir, "screenshot.png"))

    # Initialize an instance of AccessibilityTreeModifier with default settings
    modifier = AccessibilityTreeModifier(
        use_bboxes="gt",
        use_tags="gt",
        use_interact_element_text="gt",
        use_static_text="gt",
        use_ordering="raster",
        tsne_perplexity=30,
    )

    # Call the modify function
    import time

    start_time = time.time()
    modified_df = modifier.modify(gt_df, pred_df)
    modified_df.to_csv("modified_df.csv", index=False)
    print("Time taken:", time.time() - start_time)

    gt_img, _, gt_content_str, _ = draw_bounding_boxes(
        gt_df.to_csv(), screenshot_img.copy(), 1
    )
    gt_img.save("gt_img.png")
    with open("gt_content_str.txt", "w") as f:
        f.write(gt_content_str)
    pred_img, _, pred_content_str, _ = draw_bounding_boxes(
        pred_df.to_csv(), screenshot_img.copy(), 1
    )
    pred_img.save("pred_img.png")
    with open("pred_content_str.txt", "w") as f:
        f.write(pred_content_str)
    modified_img, _, modified_content_str, _ = draw_bounding_boxes(
        modified_df.to_csv(), screenshot_img.copy(), 1
    )
    modified_img.save("modified_img.png")
    with open("modified_content_str.txt", "w") as f:
        f.write(modified_content_str)

    # pd.testing.assert_frame_equal(modified_df, gt_df, check_dtype=False)
    print("done")
