import random
import requests
from typing import List
import os
import json
from pipeline.agent_pipeline import AgentPipeline
import argparse
import pandas as pd
import numpy as np
from utils.config_parser import get_config, ConfigKey, load_yaml
from utils.data_saver import DataSaver
from utils import omniact_eval
from pprint import pprint

from utils.eval import create_zeno_project, upload_zeno_project

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_env_names(environment_type) -> List[str]:
    if environment_type == "miniwob":
        with open("miniwob_rci_envs.json", "r") as f:
            env_names = json.load(f)
            # Get the name field as a list
            return [env["name"] for env in env_names][1:]
    elif environment_type == "webarena":
        config_files_dir = "webarena_config_files"
        config_files = os.listdir(config_files_dir)
        # Get all the {int}.json files in the config_files directory
        config_files = [
            os.path.join(config_files_dir, file)
            for file in config_files
            if file.endswith(".json") and file[0].isdigit()
        ]

        # It is important to go in order of the files since there are some
        # PUT or POST actions that require resetting the environment
        # Going in order prevents the need to reset.
        config_files = sorted(
            config_files, key=lambda x: int(x.split("/")[-1].split(".")[0])
        )

        filtered_config_files = []
        allowed_sites = get_config(ConfigKey.WEBSITE_TYPES)
        site_count = {}

        for config_file in config_files:
            with open(config_file, "r") as f:
                config = json.load(f)
                # if all(site in allowed_sites for site in config["sites"]):
                #     filtered_config_files.append(config_file)
                if len(config["sites"]) != 1:
                    continue
                if config["sites"][0] not in site_count:
                    site_count[config["sites"][0]] = 0

                if site_count[config["sites"][0]] < 5:
                    filtered_config_files.append(config_file)
                    site_count[config["sites"][0]] += 1

        return filtered_config_files
    elif environment_type == "vwa":
        config_files_dir = "vwa_config_files"
        # config_file_types = ["classifieds", "reddit", "shopping"]
        website_types = get_config(ConfigKey.WEBSITE_TYPES)
        for website_type in website_types:
            config_files = os.listdir(
                os.path.join(config_files_dir, f"test_{website_type}")
            )

            # Get all the {int}.json files in the config_files directory
            config_files = [
                os.path.join(config_files_dir, f"test_{website_type}", file)
                for file in config_files
                if file.endswith(".json") and file[0].isdigit()
            ]

            # It is important to go in order of the files since there are some
            # PUT or POST actions that require resetting the environment
            # Going in order prevents the need to reset.
            config_files = sorted(
                config_files, key=lambda x: int(x.split("/")[-1].split(".")[0])
            )

            filtered_config_files = []
            site_count = {}

            for config_file in config_files:
                # if "85.json" not in config_file:
                #     continue
                # else:
                #     filtered_config_files.append(config_file)
                with open(config_file, "r") as f:
                    config = json.load(f)
                    # if all(site in allowed_sites for site in config["sites"]):
                    #     filtered_config_files.append(config_file)
                    if len(config["sites"]) != 1:
                        continue

                    if (
                        config["overall_difficulty"] != "easy"
                        or config["reasoning_difficulty"] != "easy"
                        or config["visual_difficulty"] != "easy"
                    ):
                        continue

                    if config["sites"][0] not in site_count:
                        site_count[config["sites"][0]] = 0

                    if site_count[config["sites"][0]] < 1000:
                        filtered_config_files.append(config_file)
                        site_count[config["sites"][0]] += 1

            # Get random filtered_config_files
            # np.random.shuffle(filtered_config_files)
            return filtered_config_files
    elif environment_type == "omniact":
        # Create a list from 0 to 10
        # return [str(i) for i in range(3000, 3001)]
        # return [
        #     "200",
        #     "2000",
        #     "3000",
        # ]

        # env_list = sorted(
        #     json.load(open("/home/waynechi/dev/omniact/task_ids_diff_app_even.json"))
        # )

        # random.seed(get_config(ConfigKey.RANDOM_SEED))
        # env_list = sorted(
        #     [
        #         num
        #         for num in random.sample(
        #             range(2021), get_config(ConfigKey.NUM_ENVIRONMENTS)
        #         )
        #     ]
        # )
        # env_list = env_list[get_config(ConfigKey.ENV_START_IDX):]
        # TODO Finish from 1819 for the llama8b exp
        env_list = range(
            get_config(ConfigKey.ENV_START_IDX), get_config(ConfigKey.NUM_ENVIRONMENTS)
        )
        env_list = [str(num) for num in env_list]
        print(env_list)
        # save env_list to a file
        # with open("omniact_envs.json", "w") as f:
        #     json.dump(env_list, f)
        return env_list
    else:
        raise Exception("Environment Type not found")


def main(args):
    load_yaml(args.config_file)
    data_saver = DataSaver()
    data_saver.save_config()

    agent_pipeline = AgentPipeline(logger, data_saver)

    env_names = get_env_names(get_config(ConfigKey.ENVIRONMENT_TYPE))
    # load each env_name file as a json
    # env_names = ["config_files/150.json"]  # FOR TESTING WEB_ARENA

    # project = create_zeno_project(get_config(ConfigKey.ZENO_PROJECT_NAME))
    # if args.new_zeno_project:
    #     upload_zeno_project(project, env_names, np.ones(len(env_names)))

    total_reward = 0
    run_results = {}

    for env_name in env_names:
        if "classifieds" in env_name:
            # Reset the classifieds environment
            url = "http://treble.cs.cmu.edu:9980/index.php?page=reset"
            data = {"token": "4b61655535e7ed388f0d40a93600254c"}
            response = requests.post(url, data=data)

        if get_config(ConfigKey.ENVIRONMENT_TYPE) == "omniact":
            run_result = agent_pipeline.run_omniact_pipeline(
                env_name=env_name, render_mode="human"
            )
            run_results[env_name] = run_result
        else:
            run_result = agent_pipeline.run_pipeline(
                env_name=env_name, render_mode="human"
            )
            run_results[env_name] = run_result

            if run_result["reward"] > 0:
                total_reward += 1

            logger.info(
                "Total reward: {} out of: {}".format(total_reward, len(run_results))
            )

        if get_config(ConfigKey.ENVIRONMENT_TYPE) == "omniact":
            combined_gt_jsons = []
            combined_pred_jsons = []

            run_result_env_names = set(run_results.keys())
            expected_env_names = set(env_names)
            missing_env_names = expected_env_names - run_result_env_names

            # print("Missing Envs")
            # print(list(missing_env_names))
            print("Remaining: {}".format(len(missing_env_names)))

            error_env_names = []

            # Iterate through each environment name and its corresponding data
            for env_name, data in run_results.items():
                # Append the gt_json and pred_json from each environment to the combined lists
                if "gt_json" not in data or "pred_json" not in data:
                    error_env_names.append(env_name)
                else:
                    combined_gt_jsons.extend(data["gt_json"])
                    combined_pred_jsons.extend(data["pred_json"])

            print("Error Envs")
            print(error_env_names)

            total_scores = omniact_eval.get_scores(
                combined_gt_jsons, combined_pred_jsons
            )
            pprint(total_scores)

            # Save total_scores
            data_saver.save_global_key_value_pair("total_scores", total_scores)
            data_saver.save_global_key_value_pair("gt_json", combined_gt_jsons)
            data_saver.save_global_key_value_pair("pred_json", combined_pred_jsons)

    if get_config(ConfigKey.ENVIRONMENT_TYPE) == "omniact":
        return 0  # End run for omniact

    df_system = pd.DataFrame(
        {
            "output": [run_results[miniwob_env]["reward"] for miniwob_env in env_names],
            "success": [
                True if run_results[miniwob_env]["reward"] > 0 else False
                for miniwob_env in env_names
            ],
            "actions": [
                json.dumps(run_results[miniwob_env]["actions"])
                for miniwob_env in env_names
            ],
            "error": [run_results[miniwob_env]["error"] for miniwob_env in env_names],
            "error_string": [
                run_results[miniwob_env]["error_string"] for miniwob_env in env_names
            ],
        }
    )
    df_system["id"] = df_system.index
    data_saver.save_run_results_df(df_system)

    # if args.zeno_upload:
    #     project.upload_system(
    #         df_system,
    #         name=get_config(ConfigKey.EXPERIMENT_NAME),
    #         id_column="id",
    #         output_column="output",
    #     )

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Use a seaborn style for better aesthetics
    sns.set(style="whitegrid")

    df_system = df_system.sort_values(by="output", ascending=True)

    # Define a color palette
    palette = sns.color_palette(
        "viridis", len(df_system["id"])
    )  # 'viridis' is a visually appealing colormap. You can choose others like 'plasma', 'inferno', etc.

    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size as needed

    # Create a bar plot with the color palette
    ax.bar(df_system["id"], df_system["output"], color=palette, width=0.8)

    # Set x and y labels
    ax.set_xticks(df_system["id"])
    ax.set_xticklabels(
        env_names, rotation=45, ha="right", fontsize=10
    )  # Adjust fontsize as needed
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_xlabel("Environment", fontsize=12)
    ax.set_title("Reward for each environment", fontsize=14)

    # Add gridlines for better readability
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.7)

    # Adjust the layout
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--new_zeno_project",
        help="Create and upload a new zeno project with new data",
        action="store_true",
    )
    parser.add_argument(
        "--zeno_upload", help="Complete a zeno upload", action="store_true"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
