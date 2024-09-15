import pandas as pd
from bs4 import BeautifulSoup
import os
import json
from matplotlib import pyplot as plt


def compare_bounding_boxes(base_path, start, end):
    results = []
    for number in range(start, end + 1):
        pred_path = os.path.join(
            base_path, f"{number}_run_0", "step_0", "pred_bounding_boxes.csv"
        )
        gt_path = os.path.join(
            base_path, f"{number}_run_0", "step_0", "gt_bounding_boxes.csv"
        )

        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            print(f"Missing files for {number}, skipping...")
            continue

        # Load CSV files
        pred_df = pd.read_csv(pred_path)
        gt_df = pd.read_csv(gt_path)

        # Filter rows where 'interactable' is True
        pred_interactable = pred_df[pred_df["Interactable"] == True]
        gt_interactable = gt_df[gt_df["Interactable"] == True]

        # Count of interactable True in predictions and ground truth
        pred_interactable_count = len(pred_interactable)
        gt_interactable_count = len(gt_interactable)

        # Count of non-interactable rows in predictions
        pred_non_interactable_count = len(pred_df[pred_df["Interactable"] == False])

        # Store results
        results.append(
            {
                "number": number,
                "pred_interactable_count": pred_interactable_count,
                "gt_interactable_count": gt_interactable_count,
                "pred_non_interactable_count": pred_non_interactable_count,
            }
        )

    # Calculating totals and averages
    total_pred_interactable = sum(r["pred_interactable_count"] for r in results)
    total_gt_interactable = sum(r["gt_interactable_count"] for r in results)
    total_pred_non_interactable = sum(r["pred_non_interactable_count"] for r in results)

    average_pred_interactable = total_pred_interactable / len(results)
    average_gt_interactable = total_gt_interactable / len(results)
    average_pred_non_interactable = total_pred_non_interactable / len(results)

    return {
        "total_pred_interactable": total_pred_interactable,
        "total_gt_interactable": total_gt_interactable,
        "total_pred_non_interactable": total_pred_non_interactable,
        "average_pred_interactable": average_pred_interactable,
        "average_gt_interactable": average_gt_interactable,
        "average_pred_non_interactable": average_pred_non_interactable,
    }


def process_html_files(base_path):
    # Folders to search within
    folders = ["reddit_som_{}", "shopping_som_{}", "classifieds_som_{}"]
    folder_results = {"all": []}

    for folder in folders:
        results = []  # Initialize results for each folder
        for i in range(1, 4):
            folder_path = os.path.join(base_path, folder.format(i))
            category = folder.split("_")[0]
            results_path = os.path.join(folder_path, "results.json")

            with open(results_path, "r") as f:
                results_json = json.load(f)

            results_key_template = "config_files/test_{}/{}.json"

            # Walk through all files in the directory
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".html"):
                        file_path = os.path.join(root, file)
                        exp_num = file.split("_")[-1].strip(".html")
                        results_key = results_key_template.format(category, exp_num)

                        if results_key not in results_json:
                            continue

                        reward = results_json[results_key]

                        with open(file_path, "r", encoding="utf-8") as f:
                            soup = BeautifulSoup(f, "html.parser")

                            # Find all <div> with class 'state_obv' and process each <pre> within them
                            state_obv_divs = soup.find_all("div", class_="state_obv")
                            for div in state_obv_divs:
                                pre_tags = div.find_all("pre")

                                for pre in pre_tags:
                                    # Initialize counters for this pre tag
                                    count_numbered_rows = 0
                                    total_rows = 0
                                    lines = pre.get_text().split("\n")

                                    for line in lines:
                                        if line.strip():  # Ensure the line is not empty
                                            total_rows += 1
                                            if (
                                                "[" in line
                                                and "]" in line
                                                and line.split("]")[0]
                                                .strip()
                                                .startswith("[")
                                            ):
                                                try:
                                                    # Try to convert the text within first brackets to integer
                                                    int(line.split("]")[0].strip()[1:])
                                                    count_numbered_rows += 1
                                                except ValueError:
                                                    continue

                                    results.append(
                                        {
                                            "file_path": file_path,
                                            "count_numbered_rows": count_numbered_rows,
                                            "total_rows": total_rows,
                                            "reward": reward,
                                        }
                                    )

        folder_results[folder] = results
        folder_results["all"].extend(results)

    return folder_results


def summarize_results(folder_results):
    summary = {}
    for folder, results in folder_results.items():
        total_numbered_rows = sum(item["count_numbered_rows"] for item in results)
        total_rows = sum(item["total_rows"] for item in results)
        total_reward = sum(item["reward"] for item in results)
        average_numbered_rows = total_numbered_rows / len(results) if results else 0
        average_total_rows = total_rows / len(results) if results else 0
        average_reward = total_reward / len(results) if results else 0

        summary[folder] = {
            "total_numbered_rows": total_numbered_rows,
            "total_rows": total_rows,
            "total_reward": total_reward,
            "average_numbered_rows": average_numbered_rows,
            "average_total_rows": average_total_rows,
            "average_reward": average_reward,
        }
    return summary


def gather_data(folder_results):
    data = []
    for folder, results in folder_results.items():
        for result in results:
            if (
                folder != "all"
            ):  # Exclude 'all' which is just a collection of all results
                data.append(
                    {
                        "file_path": result["file_path"],
                        "count_numbered_rows": result["count_numbered_rows"],
                        "total_rows": result["total_rows"],
                        "reward": result["reward"],
                    }
                )
    return pd.DataFrame(data)


def plot_data(df, column, title):
    # Define buckets using quantiles or fixed intervals
    if column == "count_numbered_rows":
        bucket_labels = pd.cut(df[column], bins=10)
    else:  # 'total_rows'
        bucket_labels = pd.cut(df[column], bins=10)

    # Group by these buckets and calculate the average reward
    df["bucket"] = bucket_labels
    grouped = df.groupby("bucket")["reward"].mean()

    # Plotting
    plt.figure(figsize=(10, 5))
    grouped.plot(kind="bar", color="teal")
    plt.title(f"Average Reward by {title} Bucket")
    plt.xlabel("Bucket")
    plt.ylabel("Average Reward")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"bucket_{title}.png")
    plt.show()


# Define the base path
base_path = "/home/waynechi/dev/visualwebarena-dev/results/baseline_gemini_1.5"

# Process the HTML files and capture the results
folder_results = process_html_files(base_path)

# Summarize and print the results for each folder
summary = summarize_results(folder_results)
for folder, data in summary.items():
    print(f"Summary of Results for {folder}:")
    print(data)

df = gather_data(folder_results)

# Plot for both Numbered Rows and Total Rows
plot_data(df, "count_numbered_rows", "actions")
plot_data(df, "total_rows", "actions_and_static")
breakpoint()


# Base path where the folders are located
base_path = "/home/waynechi/dev/gui-agent/data/omniact_gemini_single_best_pred_bbox_all_20240512_081150"

# Get results and print summary
summary_results = compare_bounding_boxes(base_path, 0, 2020)
print(summary_results)
