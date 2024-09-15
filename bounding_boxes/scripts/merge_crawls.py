import os
import shutil


def rename_and_merge(folder1_path, folder2_path, target_folder_path):
    # Ensure target folder exists
    os.makedirs(target_folder_path, exist_ok=True)

    # List subfolders in both folders
    folder1_subfolders = [
        f
        for f in os.listdir(folder1_path)
        if os.path.isdir(os.path.join(folder1_path, f))
    ]
    folder2_subfolders = [
        f
        for f in os.listdir(folder2_path)
        if os.path.isdir(os.path.join(folder2_path, f))
    ]

    # Determine the starting page number for renaming by finding the max page number in folder 1
    max_page_num = 0
    for subfolder in folder1_subfolders:
        parts = subfolder.split("_")
        if parts[0] == "page":
            page_num = int(parts[1])
            max_page_num = max(max_page_num, page_num)

    # Copy and rename subfolders from folder 1 to target
    for subfolder in folder1_subfolders:
        shutil.copytree(
            os.path.join(folder1_path, subfolder),
            os.path.join(target_folder_path, subfolder),
        )

    # Copy and rename subfolders from folder 2 to target with updated page numbers
    for subfolder in folder2_subfolders:
        parts = subfolder.split("_")
        if parts[0] == "page":
            new_page_num = max_page_num + int(parts[1])
            new_subfolder_name = f"page_{new_page_num}_run_0"
            shutil.copytree(
                os.path.join(folder2_path, subfolder),
                os.path.join(target_folder_path, new_subfolder_name),
            )


folder_1_path = "/home/waynechi/dev/gui-agent/data/vwa_crawl_20240401_210540"
folder_2_path = "/home/waynechi/dev/gui-agent/data/vwa_crawl_20240401_230229"
target_folder_path = (
    "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/vwa_crawls/crawl_2"
)

rename_and_merge(folder_1_path, folder_2_path, target_folder_path)
