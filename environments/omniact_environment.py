import torch
import cv2
import json
import numpy as np
import os
import pandas as pd
import sys
import re
from PIL import Image

from bounding_boxes.bounding_box_utils import (
    BoundingBox,
    compute_bounding_boxes_webui,
    draw_bounding_boxes,
    get_csv_string_from_bounding_boxes,
)
from bounding_boxes.scripts.acc_tree_modifier import AccessibilityTreeModifier
from bounding_boxes.scripts.bbox_predictor import BoundingBoxPredictor
from datasets import Dataset, load_dataset
from definitions import PROJECT_DIR
from environments.environment_interface import EnvironmentInterface
from environments.set_of_mark import SoMState
from environments.task import Task
from io import BytesIO, StringIO
from ui_models import UIElementDetector
from utils.config_parser import ConfigKey, get_config
from utils import image_utils, omniact_eval
from utils.time_utils import time_function

sys.path.append(os.path.join(PROJECT_DIR, "../webui/models/screenrecognition"))


class OmniactEnvironment(EnvironmentInterface):
    _dataset = None
    _bbox_predictor = None
    _captioning_fn = None
    _base_path = os.path.expanduser("~/dev/omniact")
    _gt_boxes_blip_filename = os.path.expanduser("~/dev/omniact/gt_boxes_blip.json")
    _gt_boxes = {}

    @classmethod
    def load_dataset_once(cls, dataset_name):
        """Class method to load the dataset only once."""
        if cls._dataset is None:
            # Define paths to the processed JSON files for each dataset split
            data_files = {
                "train": os.path.join(cls._base_path, "train_processed.json"),
                "test": os.path.join(cls._base_path, "test_processed.json"),
                "val": os.path.join(cls._base_path, "val_processed.json"),
            }

            # Load the dataset
            cls._dataset = load_dataset(
                "json", data_files=data_files, split=["train", "test", "val"]
            )

        return cls._dataset

    @classmethod
    def setup_bbox_predictor(cls):
        if cls._bbox_predictor is None:
            cls._bbox_predictor = BoundingBoxPredictor(
                # model_path="/home/waynechi/dev/gui-agent/bounding_boxes/outputs/vwa_pl_100k_classes_9/model_final.pth",
                model_path=get_config(ConfigKey.BBOX_MODEL_PATH),
                cfg_path="/home/waynechi/dev/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                env="omniact",
                caption=(get_config(ConfigKey.USE_INTERACT_ELEMENT_TEXT) == "pred"),
                ocr_only=(get_config(ConfigKey.OCR_ONLY, default=False)),
            )

        return cls._bbox_predictor

    @classmethod
    def setup_captioning_fn(cls):
        if cls._captioning_fn is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            cls._captioning_fn = image_utils.get_captioning_fn(
                device, dtype, "Salesforce/blip2-flan-t5-xl"
            )

        return cls._captioning_fn

    @classmethod
    def setup_gt_boxes(cls):
        if len(cls._gt_boxes) == 0 and os.path.exists(cls._gt_boxes_blip_filename):
            with open(cls._gt_boxes_blip_filename, "r") as f:
                gt_boxes_dict = json.load(f)
                for key, value in gt_boxes_dict.items():
                    cls._gt_boxes[key] = [
                        BoundingBox.from_dict(bbox_dict) for bbox_dict in value
                    ]

    def __init__(self, env_name, logger, data_saver, render_mode="human"):
        super().__init__(logger, data_saver)

        self.dataset = self.load_dataset_once("Writer/omniact")
        self.env_name = env_name

        try:
            dataset_partition = get_config(ConfigKey.DATASET_PARTITION)
            # train is 0, test is 1, val is 2.
            if dataset_partition == "train":
                dataset_partition_key = 0
            elif dataset_partition == "test":
                dataset_partition_key = 1
            elif dataset_partition == "val":
                dataset_partition_key = 2
            else:
                raise Exception("Invalid dataset partition")
            self.sample = self.dataset[dataset_partition_key][int(env_name)]
        except:
            raise Exception("Invalid environment name")

        self.bbox_predictor = self.setup_bbox_predictor()
        self.captioning_fn = self.setup_captioning_fn()
        self.setup_gt_boxes()

        self.image_path = os.path.join(self._base_path, self.sample["image"])
        if not os.path.exists(self.image_path):
            # convert screen_1.png to screen1.png for only the last section of the path
            image_name = self.image_path.split("/")[-1]
            self.image_path = self.image_path.replace(
                image_name, image_name.replace("_", "")
            )
        self.image = Image.open(self.image_path)

        self.task_path = os.path.join(self._base_path, self.sample["task"])
        self.task, self.gt_script = self.parse_task_and_script(
            os.path.join(self._base_path, self.sample["task"])
        )
        self.task = Task(self.task, [])

        self.gt_box_path = os.path.join(self._base_path, self.sample["box"])
        if not os.path.exists(self.gt_box_path):
            gt_box_name = self.gt_box_path.split("/")[-1]
            self.gt_box_path = self.gt_box_path.replace(
                gt_box_name,
                gt_box_name.replace("_", "").replace(".json", "_boxes.json"),
            )

        self.gt_bboxes = self.parse_bounding_boxes(self.gt_box_path)

        self.acc_tree_modifier = AccessibilityTreeModifier(
            get_config(ConfigKey.USE_BBOXES),
            get_config(ConfigKey.USE_TAGS),
            get_config(ConfigKey.USE_INTERACT_ELEMENT_TEXT),
            get_config(ConfigKey.USE_STATIC_TEXT),
            get_config(ConfigKey.USE_ORDERING),
            get_config(ConfigKey.TSNE_PERPLEXITY),
        )

        self.gt_json = [
            {
                "task": self.task_path,
                "box": self.gt_box_path,
            }
        ]
        # Save the jsons
        with open("gt.json", "w") as f:
            json.dump(self.gt_json, f, indent=4)

        if get_config(ConfigKey.STATE_TYPE) == "webui_image":
            self.bounding_box_model = UIElementDetector.load_from_checkpoint(
                os.path.join(
                    PROJECT_DIR, "bounding_boxes/models/screenrecognition-web350k.ckpt"
                )
            )
            self.bounding_box_model.eval()

    def parse_task_and_script(self, filename):
        with open(filename, "r") as file:
            content = file.read()

        # Split the content into the task and script parts
        parts = content.strip().split("\n")
        task_part = parts[0]
        script_part = parts[1:]
        script_part = [
            x for x in script_part if x.lower().strip().startswith("pyautogui")
        ]
        script_part = "\n".join(script_part)

        # Further isolate the task description
        task_description = task_part.replace("Task:", "").strip()

        return task_description, script_part

    def parse_bounding_boxes(self, filename):
        if filename in self._gt_boxes:
            return self._gt_boxes[filename]
        with open(filename, "r") as file:
            data = json.load(file)

        bounding_boxes = []
        image_np = np.array(self.image)
        cv2_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        for key, entry in data.items():
            top_left = entry["top_left"]
            bottom_right = entry["bottom_right"]
            if get_config(ConfigKey.USE_INTERACT_ELEMENT_TEXT) == "gt":
                ocr_results = self.bbox_predictor.ocr_reader.readtext(cv2_image)
                cropped_image = cv2_image[
                    top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
                ]
                caption_text = ", description: {}".format(
                    self.captioning_fn([cropped_image])[0].strip()
                )
                pred_bbox = BoundingBox(
                    top=top_left[1],
                    left=top_left[0],
                    bottom=bottom_right[1],
                    right=bottom_right[0],
                )
                ocr_text, _ = self.bbox_predictor.get_ocr_text_for_bounding_box(
                    pred_bbox, ocr_results
                )

                text = "{}{}".format(ocr_text, caption_text)
            else:
                text = entry.get("label", "")

            # Instantiate BoundingBox
            bbox = BoundingBox(
                top=top_left[1],
                left=top_left[0],
                bottom=bottom_right[1],
                right=bottom_right[0],
                interactable=bool(entry.get("valid", True)),
                class_id=0,
                class_type="UI_ELEMENT",
                # text=entry.get("label", ""),
                text=text,
            )
            bounding_boxes.append(bbox)

        if filename not in self._gt_boxes:
            self._gt_boxes[filename] = bounding_boxes
            with open(self._gt_boxes_blip_filename, "w") as f:
                gt_boxes_dict = {}
                for key, value in self._gt_boxes.items():
                    gt_boxes_dict[key] = [bbox.to_dict() for bbox in value]
                json.dump(gt_boxes_dict, f, indent=4)

        return bounding_boxes

    def get_bounding_boxes(self):
        if get_config(ConfigKey.STATE_TYPE) == "som_acc_tree":
            return get_csv_string_from_bounding_boxes(self.gt_bboxes)
        else:
            raise NotImplementedError(
                "STATE_TYPE not implemented for OmniAct Environment"
            )

    def save_screenshot(self, screenshot_name):
        self.image.save(screenshot_name)

    def get_predicted_bboxes(self, pil_image: Image) -> list[list[float]]:
        # image_array = np.frombuffer(screenshot_bytes, dtype=np.uint8)
        image_np = np.array(pil_image)
        image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        csv_content = self.bbox_predictor.get_csv_string(
            image,
            score_threshold=0.3,
            captioning_fn=self.captioning_fn,
            use_all_static_text=get_config(ConfigKey.USE_ALL_STATIC_TEXT, default=True),
        )
        return csv_content

    @time_function
    def get_state(self):
        gt_som_bboxes = self.get_bounding_boxes()
        self.data_saver.save_gt_bounding_boxes(gt_som_bboxes)
        # screenshot_name = "screenshot.png"
        # self.save_screenshot(screenshot_name)
        self.data_saver.save_base_image(self.image)

        # Get Ground Truth set of marks image
        gt_som_image, gt_id2center, gt_content_str, _ = draw_bounding_boxes(
            gt_som_bboxes,
            self.image,
            1,
            img_padding=20,
            use_tag=(get_config(ConfigKey.USE_TAGS) != "none"),
            use_id=(get_config(ConfigKey.USE_IDS) != "none"),
        )
        self.data_saver.save_gt_som_image(gt_som_image)
        self.data_saver.save_key_value_pair("gt_id2center", gt_id2center)
        self.data_saver.save_key_value_pair("gt_content_str", gt_content_str)
        # gt_som_image.save("gt_som_image.png")
        gt_som_df = pd.read_csv(StringIO(gt_som_bboxes), delimiter=",", quotechar='"')
        # gt_som_df.to_csv("gt_som.csv")

        # Get predicted bboxes
        if (
            get_config(ConfigKey.USE_BBOXES) == "pred"
            and get_config(ConfigKey.USE_ORDERING) != "pred"
            and get_config(ConfigKey.USE_ORDERING) != "gt"
        ):
            # Shortcut this to save time
            pred_file_path = f"/data/waynechi/gui-agent/data/omniact_gemini_single_best_pred_bbox_all_20240512_081150/{self.env_name}_run_0/step_0/pred_bounding_boxes.csv"
            pred_som_df = pd.read_csv(pred_file_path)
            pred_som_bboxes = pred_som_df.to_csv(index=False)
        else:
            pred_som_bboxes = self.get_predicted_bboxes(self.image)
            pred_som_df = pd.read_csv(
                StringIO(pred_som_bboxes), delimiter=",", quotechar='"'
            )
        pred_som_image, pred_id2center, pred_content_str, _ = draw_bounding_boxes(
            pred_som_bboxes,
            self.image,
            1,
            img_padding=20,
            use_tag=(get_config(ConfigKey.USE_TAGS) != "none"),
            use_id=(get_config(ConfigKey.USE_IDS) != "none"),
        )
        # pred_som_df.to_csv("pred_som.csv")
        self.data_saver.save_key_value_pair("pred_id2center", pred_id2center)
        self.data_saver.save_key_value_pair("pred_content_str", pred_content_str)

        som_bboxes = self.acc_tree_modifier.modify(gt_som_df, pred_som_df)
        som_bboxes = som_bboxes.to_csv(index=False)

        add_coords = get_config(ConfigKey.ACTION_TYPES) == "omniact_pyautogui"
        bbox_img, id2center, content_str, _ = draw_bounding_boxes(
            som_bboxes,
            self.image,
            1,
            img_padding=20,
            add_coords=add_coords,
            allow_interaction_with_text=get_config(
                ConfigKey.ALLOW_INTERACTION_WITH_TEXT
            ),
            use_tag=(get_config(ConfigKey.USE_TAGS) != "none"),
            use_id=(get_config(ConfigKey.USE_IDS) != "none"),
        )
        self.data_saver.save_pred_bounding_boxes(som_bboxes)
        self.data_saver.save_pred_som_image(bbox_img)

        self.data_saver.save_key_value_pair("gt_script", self.gt_script)

        # TODO Get Predictions
        if get_config(ConfigKey.STATE_TYPE) == "image":
            return self.image
        elif get_config(ConfigKey.STATE_TYPE) == "som_acc_tree":
            return SoMState(bbox_img, content_str, id2center, None)
        else:
            raise Exception("Invalid State Type")

    def get_task(self):
        return self.task

    @time_function
    def evaluate_result(self, pred_script):
        print("Expected Result: {}".format(self.gt_script))
        print("Actual Result: {}".format(pred_script))

        self.pred_json = [pred_script]

        with open("pred.json", "w") as f:
            json.dump(self.pred_json, f, indent=4)

        score_json = omniact_eval.get_scores(self.gt_json, self.pred_json)

        return score_json

    def execute_action(self, action, step_id):
        raise NotImplementedError("Omniact doesn't execute actions")

    def clean_up(self):
        pass
