from urllib.parse import urlparse

from typing import List
import requests
from environments.task import Task
import json
import os
from definitions import WEB_ARENA_DIR

import subprocess
from environments.environment_interface import EnvironmentInterface
from actions.action import Action
from PIL import Image as PILImage
from PIL.Image import Image
from io import BytesIO
import time
from bounding_boxes.model_client import ModelClient


import gymnasium
from utils.config_parser import get_config, ConfigKey
from bounding_boxes.bounding_box_utils import draw_bounding_boxes
from environments.set_of_mark import SoMState

# Init an environment
from browser_env import (
    Action as WebArenaAction,
    ActionTypes as WebArenaActionTypes,
    Trajectory,
    ScriptBrowserEnv,
    create_mouse_click_action,
    create_keyboard_type_action,
    create_key_press_action,
    create_scroll_action,
    create_stop_action,
    create_none_action,
)


class WebArenaEnvironment(EnvironmentInterface):
    def __init__(self, env_name, logger, data_saver, render_mode="human"):
        super().__init__(logger, data_saver)
        self.env_name = env_name
        # For webarena, the env_name leads to a config file
        assert os.path.exists(self.env_name)

        from evaluation_harness.evaluators import evaluator_router

        self.model_client = ModelClient()
        self.evaluator = evaluator_router(
            self.env_name, captioning_fn=self.captioning_fn
        )

        if not os.path.exists(".auth"):
            subprocess.run(["bash", "web_arena_prepare.sh"])

        # Init the environment
        self.env = ScriptBrowserEnv(
            headless=(render_mode != "human"),
            slow_mo=0,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 2048},
        )

        self.trajectory: Trajectory = []

        self.observation, self.info = self.env.reset(
            options={"config_file": self.env_name}
        )
        with open(self.env_name, "r") as f:
            config = json.load(f)
            if (
                "image" in config
                and config["image"] is not None
                and len(config["image"]) > 0
            ):
                if isinstance(config["image"], str):
                    images = [self.open_image_from_str(config["image"])]
                elif isinstance(config["image"], list):
                    image_str_list = config["image"]
                    images = [
                        self.open_image_from_str(image_str)
                        for image_str in image_str_list
                    ]
                else:
                    raise Exception("Invalid image config")
            else:
                images = None
            self.task = Task(config["intent"], images)

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])

    def captioning_fn(self, images: List[Image], prompts: List[str]):
        return [
            self.model_client.query_model(image, prompt)["answer"]
            for image, prompt in zip(images, prompts)
        ]

    def get_window_bounds(self):
        window_bounds = {
            "upper_bound": self.env.observation_handler.image_processor.browser_config[
                "win_upper_bound"
            ],
            "lower_bound": self.env.observation_handler.image_processor.browser_config[
                "win_lower_bound"
            ],
            "left_bound": self.env.observation_handler.image_processor.browser_config[
                "win_left_bound"
            ],
            "right_bound": self.env.observation_handler.image_processor.browser_config[
                "win_right_bound"
            ],
        }

        return window_bounds

    def open_image_from_str(self, image_str):
        if self.is_valid_url(image_str):
            return self.open_image_from_url(image_str)
        else:
            image = PILImage.open(image_str)
            if image.format != "PNG":
                image = image.convert("RGBA")
            return image

    def open_image_from_url(self, url):
        # Send a GET request to the URL
        response = requests.get(url)

        # Ensure the request was successful
        response.raise_for_status()

        # Open the image
        image = PILImage.open(BytesIO(response.content))

        # Convert to PNG if not already in PNG format
        if image.format != "PNG":
            image = image.convert("RGBA")  # Convert image to PNG format

        return image

    def get_viewport_size(self):
        return self.env.page.viewport_size

    def get_bounding_boxes(self):
        js_script = """
        (() => {
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
        })();
        """
        bounding_boxes = self.env.page.evaluate(js_script)
        return bounding_boxes

    def save_screenshot(self, screenshot_name):
        self.env.page.screenshot(path=screenshot_name)

    def get_pixel_ratio(self):
        return self.env.page.evaluate("window.devicePixelRatio")

    def get_state(self):
        self.data_saver.save_dom(self.observation)
        gt_bounding_boxes_csv = self.get_bounding_boxes()
        self.data_saver.save_gt_bounding_boxes(gt_bounding_boxes_csv)
        pixel_ratio = self.get_pixel_ratio()
        screenshot_name = "state.png"
        self.save_screenshot(screenshot_name)

        image = PILImage.open(screenshot_name)
        self.data_saver.save_base_image(image)
        gt_som_image, id2center, gt_content_str, _ = draw_bounding_boxes(
            gt_bounding_boxes_csv,
            image,
            pixel_ratio,
            viewport_size=self.get_viewport_size(),
            window_bounds=self.get_window_bounds(),
        )
        self.data_saver.save_gt_som_image(gt_som_image)

        if get_config(ConfigKey.STATE_TYPE) == "dom":
            return self.observation
        elif get_config(ConfigKey.STATE_TYPE) == "dom_image":
            return SoMState(
                gt_som_image,
                gt_content_str,
                id2center,
                self.observation,
                self.get_page_url(),
            )
        elif get_config(ConfigKey.STATE_TYPE) == "webui_image":
            raise NotImplementedError("WebUI Image not implemented yet")
        else:
            raise Exception("Invalid State Type")

    def get_page_url(self):
        return self.info["page"].url

    def get_task(self):
        return self.task

    def action_to_web_arena_action(self, action: Action) -> List[WebArenaAction]:
        viewport_size = self.get_viewport_size()
        viewport_width = viewport_size["width"]
        viewport_height = viewport_size["height"]
        if action.action_type == "CLICK_COORDS":
            return [
                create_mouse_click_action(
                    action.action_params["coords"][0] / viewport_width,
                    action.action_params["coords"][1] / viewport_height,
                )
            ]
        elif action.action_type == "TYPE_TEXT":
            actions = [
                create_mouse_click_action(
                    action.action_params["coords"][0] / viewport_width,
                    action.action_params["coords"][1] / viewport_height,
                ),
                create_keyboard_type_action(action.action_params["text"]),
            ]
            if action.action_params["press_enter"]:
                actions.append(create_key_press_action("Enter"))
            return actions
        elif action.action_type == "SCROLL":
            return [create_scroll_action(action.action_params["direction"])]
        elif action.action_type == "STOP":
            return [create_stop_action(action.action_params["answer"])]
        elif action.action_type == "NONE":
            return [create_none_action()]

        raise Exception("No action found")

    def execute_action(self, action: Action, step_id: int):
        try:
            web_arena_actions = self.action_to_web_arena_action(action)
            for web_arena_action in web_arena_actions:
                self.observation, reward, terminated, truncated, self.info = (
                    self.env.step(web_arena_action)
                )
                state_info = {"observation": self.observation, "info": self.info}
                self.trajectory.append(state_info)
                self.trajectory.append(web_arena_action)
                # TODO Draw where I am clicking for debug purposes
                if action.action_type == "CLICK_COORDS":
                    self.draw_and_save_click(action.action_params["coords"])

                if web_arena_action["action_type"] == WebArenaActionTypes.STOP:
                    terminated = True
        except Exception as e:
            self.logger.error("Error executing action: {}".format(action))
            raise e

        if step_id >= get_config(ConfigKey.MAX_STEPS) - 1:
            terminated = True

        if terminated:
            reward = self.evaluator(
                trajectory=self.trajectory,
                config_file=self.env_name,
                page=self.env.page,
                client=self.env.get_page_client(self.env.page),
            )
        else:
            reward = 0

        return reward, terminated

    def clean_up(self):
        self.env.close()
