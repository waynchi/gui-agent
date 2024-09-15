import pandas as pd
from io import StringIO
import pickle
import json
import yaml
import os
from pathlib import Path
from utils.config_parser import get_config, get_yaml_data, ConfigKey
from pandas import DataFrame
from PIL import Image
from datetime import datetime
from pipeline.run_result import RunResult
from typing import Dict


class DataSaver:
    def __init__(self):
        # Setting up the various folders
        filename_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = os.path.join(
            "/data/waynechi/gui-agent/data",
            "{}_{}".format(get_config(ConfigKey.EXPERIMENT_NAME), filename_timestamp),
        )
        self.create_data_path(data_path)

    def create_data_path(self, data_path):
        Path(data_path).mkdir(parents=True, exist_ok=True)

        self.data_path = data_path

    def save_config(self):
        # write yaml to exp data path
        yaml_data, default_yaml_data = get_yaml_data()
        with open(os.path.join(self.data_path, "config.yaml"), "w") as file:
            yaml.dump(yaml_data, file)
        with open(os.path.join(self.data_path, "default_config.yaml"), "w") as file:
            yaml.dump(default_yaml_data, file)

    def save_run_results_df(self, df: DataFrame):
        # write run results to exp data path
        df.to_csv(os.path.join(self.data_path, "run_results.csv"))

    def start_run(self, env_name, run_id, task):
        # write env name to exp data path
        self.run_path = os.path.join(
            self.data_path, "{}_run_{}".format(env_name, run_id)
        )
        Path(self.run_path).mkdir(parents=True, exist_ok=False)
        with open(os.path.join(self.run_path, "task.txt"), "w") as file:
            file.write(task.task)

    def save_run_result(self, run_result: RunResult):
        # write run result to exp data path
        try:
            with open(os.path.join(self.run_path, "run_result.json"), "w") as file:
                json.dump(run_result, file, indent=4)
        except:
            print("Error saving run result")

    def start_step(self, step_id):
        # write step id to exp data path
        self.step_path = os.path.join(self.run_path, "step_{}".format(step_id))
        Path(self.step_path).mkdir(parents=True, exist_ok=False)
        self.base_image: Image = None
        self.gt_som_image: Image = None
        self.pred_som_image: Image = None
        self.action_image: Image = None
        self.prompts = None
        self.response = None
        self.dom = None
        self.info = {}

    def save_base_image(self, image: Image):
        # save base image to step path
        if self.base_image is not None:
            raise Exception("Base image already saved. Start new step before saving.")
        self.base_image = image
        image.save(os.path.join(self.step_path, "base.png"))

    def save_gt_som_image(self, image: Image):
        # save gt som image to step path
        if self.gt_som_image is not None:
            raise Exception("GT som image already saved. Start new step before saving.")
        self.gt_som_image = image
        image.save(os.path.join(self.step_path, "gt_som.png"))

    def save_gt_bounding_boxes(self, bounding_boxes_csv):
        # Check if gt bounding boxes are already saved
        filename = os.path.join(self.step_path, "gt_bounding_boxes.csv")
        if os.path.exists(filename):
            raise Exception(
                "GT bounding boxes already saved. Start new step before saving."
            )
        df = pd.read_csv(StringIO(bounding_boxes_csv), delimiter=",", quotechar='"')
        df.to_csv(filename, index=False)

    def save_pred_bounding_boxes(self, bounding_boxes_csv):
        filename = os.path.join(self.step_path, "pred_bounding_boxes.csv")
        if os.path.exists(filename):
            raise Exception(
                "Pred bounding boxes already saved. Start new step before saving."
            )
        df = pd.read_csv(StringIO(bounding_boxes_csv), delimiter=",", quotechar='"')
        df.to_csv(filename, index=False)

    def save_pred_som_image(self, image: Image):
        if self.pred_som_image is not None:
            raise Exception(
                "Pred som image already saved. Start new step before saving."
            )
        self.pred_som_image = image
        # save pred som image to step path
        image.save(os.path.join(self.step_path, "pred_som.png"))

    def save_click_image(self, image: Image):
        if self.action_image is not None:
            raise Exception("Action image already saved. Start new step before saving.")
        self.action_image = image
        # save action to step path
        image.save(os.path.join(self.step_path, "action.png"))

    def remove_images_from_messages(self, messages):
        filtered_messages = []
        for message in messages:
            if isinstance(message, str):
                filtered_messages.append(message)
        return filtered_messages

    def save_filter_prompt_strings(self, prompts):
        # save prompt to step path
        if get_config(ConfigKey.PROVIDER) == "google":
            # Remove all of the image prompts
            prompts = self.remove_images_from_messages(prompts)

        self.info["filter_prompt"] = prompts
        # Write info json
        with open(os.path.join(self.step_path, "info.json"), "w") as file:
            json.dump(self.info, file, indent=4)

    def save_prompt_strings(self, prompts):
        # save prompt to step path
        if self.prompts is not None:
            raise Exception("Prompt already saved. Start new step before saving.")

        if get_config(ConfigKey.PROVIDER) == "google":
            # Remove all of the image prompts
            prompts = self.remove_images_from_messages(prompts)

        self.prompts = prompts
        self.info["prompt"] = prompts
        # Write info json
        with open(os.path.join(self.step_path, "info.json"), "w") as file:
            json.dump(self.info, file, indent=4)

    def save_prompts(self, prompts):
        # save prompt to step path
        if self.prompts is not None:
            raise Exception("Prompt already saved. Start new step before saving.")
        self.prompts = prompts
        self.info["prompt"] = [str(prompt) for prompt in prompts]
        # Write info json
        with open(os.path.join(self.step_path, "info.json"), "w") as file:
            json.dump(self.info, file, indent=4)

        counter = 0
        if len(prompts[-1].captioned_images) > 0:
            for captioned_image in prompts[-1].captioned_images:
                captioned_image.image.save(
                    os.path.join(
                        self.step_path,
                        "prompt_{}.png".format(counter),
                    )
                )
                counter += 1

    def save_response(self, response):
        # save response to step path
        if self.response is not None:
            raise Exception("Response already saved. Start new step before saving.")
        self.response = response
        self.info["response"] = response
        with open(os.path.join(self.step_path, "info.json"), "w") as file:
            json.dump(self.info, file, indent=4)

    def save_info(self, info: Dict):
        self.info.update(info)
        with open(os.path.join(self.step_path, "info.json"), "w") as file:
            json.dump(self.info, file, indent=4)

    def save_dom(self, dom):
        if self.dom is not None:
            raise Exception("DOM already saved. Start new step before saving.")

        self.dom = dom
        with open(os.path.join(self.step_path, "dom.pkl"), "wb") as file:
            pickle.dump(self.dom, file)

    def save_key_value_pair(self, key, value):
        # save gt id2center to step path
        self.info[key] = value
        with open(os.path.join(self.step_path, "info.json"), "w") as file:
            json.dump(self.info, file, indent=4)

    def save_global_key_value_pair(self, key, value):
        with open(os.path.join(self.run_path, f"{key}.json"), "w") as file:
            json.dump(value, file, indent=4)
