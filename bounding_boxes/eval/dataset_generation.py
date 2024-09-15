import shutil
import os
import json
from io import StringIO
import miniwob
import gymnasium
from definitions import PROJECT_DIR
import pandas as pd
from bounding_boxes.bounding_box_utils import (
    BoundingBox,
    get_valid_bounding_boxes_from_csv_string,
)
from environments.webarena_environment import WebArenaEnvironment
from bounding_boxes.bounding_box_utils import draw_bounding_boxes
from PIL import Image

MINIWOB_DATASET_PATH = os.path.join(
    PROJECT_DIR, "bounding_boxes", "datasets", "miniwob"
)
WEBARENA_DATASET_PATH = os.path.join(
    PROJECT_DIR, "bounding_boxes", "datasets", "webarena"
)
COMCRAWL_DATASET_PATH = os.path.join(
    PROJECT_DIR, "bounding_boxes", "datasets", "comcrawl"
)
VWA_DATASET_PATH = os.path.join(PROJECT_DIR, "bounding_boxes", "datasets", "vwa_2")
VWA_HTML_DATASET_PATH = os.path.join(
    PROJECT_DIR, "bounding_boxes", "datasets", "vwa_html"
)


def generate_static_miniwob_dataset():
    with open(os.path.join(PROJECT_DIR, "miniwob_rci_envs.json"), "r") as f:
        miniwob_envs = json.load(f)
        miniwob_envs = [env["name"] for env in miniwob_envs][:]

    sample_per_env = 10

    env_dfs = []
    for env_name in miniwob_envs:
        env = gymnasium.make("miniwob/{}-v1".format(env_name), render_mode="human")
        for i in range(sample_per_env):
            env.reset()
            csv_string = env.unwrapped.get_bounding_boxes()
            screenshot_name = os.path.join(
                MINIWOB_DATASET_PATH, "images", "{}-{}.png".format(env_name, i)
            )
            env.unwrapped.instance.driver.save_screenshot(screenshot_name)
            image = Image.open(screenshot_name).convert("RGB")

            # Crop the image to the left-hand third
            width, height = image.size
            image = image.crop((0, 0, width // 3, height))
            image.save(screenshot_name)

            bounding_boxes = get_valid_bounding_boxes_from_csv_string(csv_string)
            env_df = pd.DataFrame(
                [bounding_box.to_dict() for bounding_box in bounding_boxes]
            )
            env_df["environment"] = "{}-{}".format(env_name, i)
            env_df["box_index"] = env_df.index
            env_df["image"] = screenshot_name
            env_dfs.append(env_df)

    # combine list of dfs into one df
    df = pd.concat(env_dfs)
    df.to_csv(
        os.path.join(MINIWOB_DATASET_PATH, "static_miniwob_dataset.csv"), index=False
    )


def generate_static_webarena_dataset():
    # loop through all files in config_files
    config_files_dir = "config_files"
    config_files = os.listdir(config_files_dir)
    env_dfs = []
    sample_per_env = 1  # Webarena seems static so don't bother with this?
    for config_file in config_files:
        print("config_file: ", config_file)
        for i in range(sample_per_env):
            web_arena_environment = WebArenaEnvironment(
                os.path.join(config_files_dir, config_file), None, render_mode="human"
            )
            csv_string = web_arena_environment.get_bounding_boxes()
            env_name = "env-{}".format(config_file.split(".")[0])
            screenshot_name = os.path.join(
                WEBARENA_DATASET_PATH, "images", "{}-{}.png".format(env_name, i)
            )
            web_arena_environment.save_screenshot(screenshot_name)

            bounding_boxes = get_valid_bounding_boxes_from_csv_string(csv_string)
            env_df = pd.DataFrame(
                [bounding_box.to_dict() for bounding_box in bounding_boxes]
            )
            env_df["environment"] = "{}-{}".format(env_name, i)
            env_df["box_index"] = env_df.index
            env_df["image"] = screenshot_name
            env_dfs.append(env_df)

            web_arena_environment.clean_up()
            # pixel_ratio = web_arena_environment.get_pixel_ratio()
            # image = Image.open(screenshot_name)
            # som_image, id2center, content_str, _ = draw_bounding_boxes(csv_string, image, pixel_ratio)
            # som_image.save("test.png")
            # breakpoint()

    # combine list of dfs into one df
    df = pd.concat(env_dfs)
    df.to_csv(
        os.path.join(WEBARENA_DATASET_PATH, "static_webarena_dataset.csv"), index=False
    )


def generate_static_dataset_from_run_data(
    folder, sub_folder, extra_string="", is_vwa=False
):
    # loop through all files in config_files
    env_dfs = []
    # Loop through the directory
    # Website I'm keeping: Youtube, Amazon, Ebay, Reddit
    websites_to_skip = [
        "robots.txt",
        "google.com",
        "facebook.com",
        "instagram.com",
        "pinterest.com",
        "twitter.com",
        "fandom.com",
        "quora.com",
        "linkedin.com",
    ]
    site_type_count = {
        "reddit": 0,
        "classifieds": 0,
        "shopping": 0,
    }  # This is for vwa

    websites_visited = set()
    images_folder = os.path.join(folder, sub_folder, "images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    for webpage_folder in os.listdir(os.path.join(folder, sub_folder)):
        if not os.path.isdir(os.path.join(folder, sub_folder, webpage_folder)):
            continue
        info_json_path = os.path.join(
            folder, sub_folder, webpage_folder, "step_0", "info.json"
        )
        if not os.path.exists(info_json_path):
            continue

        with open(info_json_path, "r") as f:
            info_json = json.load(f)
            if info_json["url"] not in websites_visited:
                websites_visited.add(info_json["url"])

                if is_vwa:
                    if "reddit" in info_json["base_config_file"]:
                        site_type_count["reddit"] += 1
                        if site_type_count["reddit"] > 200:
                            continue
                    elif "classifieds" in info_json["base_config_file"]:
                        site_type_count["classifieds"] += 1
                        if site_type_count["classifieds"] > 200:
                            continue
                    elif "shopping" in info_json["base_config_file"]:
                        site_type_count["shopping"] += 1
                        if site_type_count["shopping"] > 200:
                            continue
            else:
                continue

            # Special condition for comcrawl
            if any(
                website_to_skip in info_json["url"]
                for website_to_skip in websites_to_skip
            ):
                continue
        for step_folder in os.listdir(os.path.join(folder, sub_folder, webpage_folder)):
            step_folder_full = os.path.join(
                folder, sub_folder, webpage_folder, step_folder
            )
            if not os.path.isdir(step_folder_full):
                continue

            try:
                df = pd.read_csv(
                    os.path.join(step_folder_full, "gt_bounding_boxes.csv")
                )
            except FileNotFoundError:
                continue
            if df.empty:
                continue

            with open(os.path.join(step_folder_full, "info.json"), "r") as f:
                info_json = json.load(f)

            output = StringIO()
            df.to_csv(output, index=False, sep=",")
            csv_string = output.getvalue()
            env_name = "{}-{}-{}".format(webpage_folder, step_folder, info_json["url"])
            screenshot_name = os.path.join(step_folder_full, "base.png")
            # Copy screenshot over to images folder
            screenshot_name_new = os.path.join(
                images_folder, "{}-{}.png".format(webpage_folder, step_folder)
            )
            shutil.copyfile(screenshot_name, screenshot_name_new)

            bounding_boxes = get_valid_bounding_boxes_from_csv_string(
                csv_string,
                x_offset=info_json["window_bounds"]["left_bound"],
                y_offset=info_json["window_bounds"]["upper_bound"],
            )

            env_df = pd.DataFrame(
                [bounding_box.to_dict() for bounding_box in bounding_boxes]
            )
            env_df["environment"] = "{}-{}".format(env_name, 0)
            env_df["box_index"] = env_df.index
            env_df["image"] = screenshot_name_new
            env_dfs.append(env_df)

    # combine list of dfs into one df
    df = pd.concat(env_dfs)
    df.to_csv(
        os.path.join(
            folder, "static_{}{}_dataset.csv".format(sub_folder, extra_string)
        ),
        index=False,
    )


def generate_static_dataset_from_run_data_comcrawl(
    data_folder, image_folder, output_filename
):
    # loop through all files in config_files
    env_dfs = []
    # Loop through the directory
    websites_visited = set()
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    for webpage_folder_name in os.listdir(data_folder):
        webpage_folder_path = os.path.join(data_folder, webpage_folder_name)
        if not os.path.isdir(webpage_folder_path):
            continue
        info_json_path = os.path.join(webpage_folder_path, "step_0", "info.json")
        if not os.path.exists(info_json_path):
            continue

        with open(info_json_path, "r") as f:
            info_json = json.load(f)
            if info_json["base_url"] not in websites_visited:
                websites_visited.add(info_json["base_url"])
            else:
                continue

        for step_folder_name in os.listdir(webpage_folder_path):
            step_folder_path = os.path.join(webpage_folder_path, step_folder_name)
            if not os.path.isdir(step_folder_path):
                continue

            try:
                df = pd.read_csv(
                    os.path.join(step_folder_path, "gt_bounding_boxes.csv")
                )
            except FileNotFoundError:
                continue
            if df.empty:
                continue

            with open(os.path.join(step_folder_path, "info.json"), "r") as f:
                info_json = json.load(f)

            output = StringIO()
            df.to_csv(output, index=False, sep=",")
            csv_string = output.getvalue()
            env_name = "{}_{}".format(info_json["url"], step_folder_name)
            screenshot_name = os.path.join(step_folder_path, "base.png")
            # Copy screenshot over to images folder
            screenshot_name_new = os.path.join(
                image_folder, "{}_{}.png".format(webpage_folder_name, step_folder_name)
            )
            shutil.copyfile(screenshot_name, screenshot_name_new)

            bounding_boxes = get_valid_bounding_boxes_from_csv_string(
                csv_string,
                x_offset=info_json["window_bounds"]["left_bound"],
                y_offset=info_json["window_bounds"]["upper_bound"],
            )

            env_df = pd.DataFrame(
                [bounding_box.to_dict() for bounding_box in bounding_boxes]
            )
            env_df["environment"] = "{}_{}".format(env_name, 0)
            env_df["box_index"] = env_df.index
            env_df["image"] = screenshot_name_new
            env_dfs.append(env_df)

    # combine list of dfs into one df
    df = pd.concat(env_dfs)
    df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    # generate_static_miniwob_dataset()
    # generate_static_webarena_dataset()
    # generate_static_dataset_from_run_data(COMCRAWL_DATASET_PATH, "comcrawl_no_btn_1", extra_string="_classes_2")
    # generate_static_dataset_from_run_data(
    #     VWA_HTML_DATASET_PATH, "vwa_crawl", is_vwa=False, extra_string="_classes_9"
    # )
    generate_static_dataset_from_run_data_comcrawl(
        "/home/waynechi/dev/gui-agent/data/comcrawl_2_large_part_7",
        "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/comcrawl_2_large/images_7",
        "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/comcrawl_2_large/dataset_7.csv",
    )
