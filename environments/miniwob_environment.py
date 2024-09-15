import os
from definitions import PROJECT_DIR
from environments.environment_interface import EnvironmentInterface
from actions.action import Action
from PIL import Image
import miniwob
import sys

sys.path.append(os.path.join(PROJECT_DIR, "../webui/models/screenrecognition"))

import gymnasium
from utils.config_parser import get_config, ConfigKey
from bounding_boxes.bounding_box_utils import draw_bounding_boxes
from environments.set_of_mark import SoMState
from bounding_boxes.bounding_box_utils import (
    compute_bounding_boxes_webui,
    get_csv_string_from_bounding_boxes,
)
from ui_models import UIElementDetector


class MiniWobEnvironment(EnvironmentInterface):
    def __init__(self, env_name, logger, data_saver, render_mode="human"):
        super().__init__(logger, data_saver)
        self.env = gymnasium.make(
            env_name,
            render_mode=render_mode,
            # reward_processor=get_binary_reward,
            # action_space_config="humphreys22"
        )
        self.env.unwrapped.action_space_config.coord_bins = None
        self.observation, self.info = self.env.reset()
        if get_config(ConfigKey.STATE_TYPE) == "webui_image":
            self.bounding_box_model = UIElementDetector.load_from_checkpoint(
                os.path.join(
                    PROJECT_DIR, "bounding_boxes/models/screenrecognition-web350k.ckpt"
                )
            )
            self.bounding_box_model.eval()

    def get_bounding_boxes(self):
        js_script = """
            const interactableSelectors = [
                'a[href]:not(:has(img))', 'a[href] img', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
                '[tabindex]:not([tabindex="-1"])', '[contenteditable="true"]', '[role="button"]', '[role="link"]',
                '[role="checkbox"]', '[role="menuitem"]', '[role="tab"]', '[draggable="true"]'
            ];

            const textSelectors = ['p', 'span', 'div:not(:has(*))', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article'];
            const modifiedTextSelectors = textSelectors.map(selector =>
                `:not(${interactableSelectors.join(', ')}):not(style) > ${selector}`
            );

            const combinedSelectors = [...interactableSelectors, ...modifiedTextSelectors];
            const elements = document.querySelectorAll(combinedSelectors.join(', '));

            const pixelRatio = window.devicePixelRatio;
            let csvContent = "ID,Element,Top,Right,Bottom,Left,Width,Height,Alt,Class,Id,TextContent,Interactable\\n";
            let counter = 1;

            elements.forEach(element => {
                const rect = element.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                let altText = element.getAttribute('alt') || '';
                altText = altText.replace(/"/g, ''); // Escape double quotes in alt text
                const classList = element.className || '';
                const id = element.id || '';
                let textContent = element.textContent || '';
                textContent = textContent.replace(/"/g, ''); // Escape double quotes in textContent

                // Determine if the element is interactable
                const isInteractable = interactableSelectors.some(selector => element.matches(selector));

                const dataString = [
                    counter, element.tagName, (rect.top + window.scrollY) * pixelRatio,
                    (rect.right + window.scrollX) * pixelRatio, (rect.bottom + window.scrollY) * pixelRatio,
                    (rect.left + window.scrollX) * pixelRatio, rect.width * pixelRatio, rect.height * pixelRatio,
                    altText, classList, id, textContent, isInteractable
                ].map(value => `"${value}"`).join(",");

                csvContent += dataString + "\\n";
                counter++;
            });

            return csvContent;
        """
        bounding_boxes = self.env.unwrapped.instance.driver.execute_script(js_script)

        return bounding_boxes

    def get_pixel_ratio(self):
        return self.env.unwrapped.instance.driver.execute_script(
            "return window.devicePixelRatio;"
        )

    def save_screenshot(self, screenshot_name):
        self.env.unwrapped.instance.driver.save_screenshot(screenshot_name)

    def get_state(self):
        self.data_saver.save_dom(self.observation["dom_elements"])

        gt_bounding_boxes_csv = self.get_bounding_boxes()
        self.data_saver.save_gt_bounding_boxes(gt_bounding_boxes_csv)
        pixel_ratio = self.get_pixel_ratio()
        screenshot_name = self.temp_state_filename
        self.save_screenshot(screenshot_name)

        # Get set of marks image
        image = Image.open(screenshot_name)
        self.data_saver.save_base_image(image)
        gt_som_image, gt_id2center, gt_content_str, _ = draw_bounding_boxes(
            gt_bounding_boxes_csv, image, pixel_ratio, img_padding=20
        )
        self.data_saver.save_gt_som_image(gt_som_image)
        self.data_saver.save_key_value_pair("gt_id2center", gt_id2center)

        if get_config(ConfigKey.STATE_TYPE) == "dom":
            return self.observation
        elif get_config(ConfigKey.STATE_TYPE) == "dom_image":
            return SoMState(
                gt_som_image, gt_content_str, gt_id2center, self.observation
            )
        elif get_config(ConfigKey.STATE_TYPE) == "webui_image":
            bounding_boxes_dict = compute_bounding_boxes_webui(
                self.bounding_box_model,
                screenshot_name,
                confidence_threshold=get_config(ConfigKey.CONFIDENCE_THRESHOLD),
                merge_iou_thresholds=[get_config(ConfigKey.MERGE_IOU_THRESHOLD)],
            )
            pred_bounding_boxes_csv = get_csv_string_from_bounding_boxes(
                bounding_boxes_dict[get_config(ConfigKey.MERGE_IOU_THRESHOLD)]
            )
            self.data_saver.save_pred_bounding_boxes(pred_bounding_boxes_csv)
            pred_som_image, pred_id2center, pred_content_str, _ = draw_bounding_boxes(
                pred_bounding_boxes_csv, image, pixel_ratio
            )
            self.data_saver.save_pred_som_image(pred_som_image)
            self.data_saver.save_key_value_pair("pred_id2center", pred_id2center)
            return SoMState(
                pred_som_image, pred_content_str, pred_id2center, self.observation
            )
        else:
            raise Exception("Invalid State Type")

    def get_task(self):
        return self.observation["utterance"]

    def execute_action(self, action: Action, step_id: int):
        try:
            env_action = self.env.unwrapped.create_action(
                miniwob.action.ActionTypes(action.action_type), **action.action_params
            )
            if action.action_type == "CLICK_COORDS":
                self.draw_and_save_click(action.action_params["coords"])

            self.logger.debug(self.observation["dom_elements"])
            self.observation, reward, terminated, truncated, info = self.env.step(
                env_action
            )
            self.logger.debug(self.observation["dom_elements"])

        except Exception as e:
            self.logger.error("Error executing action: {}".format(action))
            raise e

        if step_id >= get_config(ConfigKey.MAX_STEPS) - 1:
            terminated = True

        return reward, terminated

    def clean_up(self):
        self.env.close()
