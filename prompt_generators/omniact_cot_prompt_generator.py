import json
import re
from environments.set_of_mark import SoMState
from prompt_generators.prompt_generator_interface import (
    PromptGeneratorInterface,
)
from PIL import Image
from typing import List
from prompt_generators.prompt import Prompt
from transformers import GPT2Tokenizer
from utils.config_parser import get_config, ConfigKey
from utils.time_utils import time_function
from io import BytesIO
from environments.task import Task
from prompt_generators.tokenizers import Tokenizer
import base64
from agents.omniact_agent import OmniactAgent

try:
    from vertexai.preview.generative_models import Image as VertexImage
except:
    print(
        "Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image"
    )


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def pil_to_vertex(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_vertex = VertexImage.from_bytes(byte_data)
    return img_vertex


class OmniactCotPromptGenerator(PromptGeneratorInterface):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        environment,
        logger,
        data_saver,
        instruction_path,
        omniact_agent: OmniactAgent,
    ):
        super().__init__(environment, logger)
        self.instruction = json.load(open(instruction_path))
        self.tokenizer = Tokenizer(
            get_config(ConfigKey.PROVIDER), get_config(ConfigKey.MODEL_NAME)
        )
        self.data_saver = data_saver
        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.no_img = get_config(ConfigKey.NO_IMG)
        self.omniact_agent = omniact_agent

    @time_function
    def generate_prompts(self, state: SoMState, task: Task) -> List[str]:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]

        obs = state.acc_tree

        max_obs_length = get_config(ConfigKey.TOKEN_LIMIT)
        if max_obs_length:
            obs = self.truncate_to_token_limit(obs, self.tokenizer, max_obs_length)

        if "filter" in get_config(ConfigKey.PROMPT_MODE):
            obs = self.filter_obs(obs, task.task, state.som_image)

        current = template.format(
            objective=task.task,
            observation=obs,
        )

        prompts = self.get_lm_api_input(intro, examples, current, state.som_image, [])
        return prompts

    def filter_obs(self, obs: str, task: str, page_screenshot_img) -> str:
        """
        Pre-filter observation as with omniact
        """
        if get_config(ConfigKey.PROVIDER) == "openai":
            raise NotImplementedError("Filtering not implemented for OpenAI")
        elif get_config(ConfigKey.PROVIDER) == "google":
            filter_instruction = json.load(
                open(get_config(ConfigKey.FILTER_INSTRUCTION_PATH))
            )
            intro = filter_instruction["intro"]
            examples = filter_instruction["examples"]
            example_img = Image.open(examples[3])
            message = [
                f"""{intro}
Sample Task:
{examples[0]}
Sample UI Elements:
{examples[1]}
Sample Filtered UI Elements:
{examples[2]}
IMAGE (1): Example Image
\n\n
Given Task:
{task}
Given UI Elements:
{obs}
IMAGE (2): Given Image
""",
                pil_to_vertex(example_img),
                pil_to_vertex(page_screenshot_img),
            ]

        self.data_saver.save_filter_prompt_strings(message)
        response = self.omniact_agent.get_response(message)
        action_splitter = "```"
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)

        elements = match.group(1).strip().split("\n")
        elements = [x.strip() for x in elements]
        filtered_obs = "\n".join(elements)
        self.data_saver.save_key_value_pair("filtered_obs", filtered_obs)

        return filtered_obs

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        current: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
    ) -> List[str]:
        """Return the require format for an API"""
        message: list[dict[str, str]] | str | list[str | Image.Image]
        # GPT4 vs Gemini
        if get_config(ConfigKey.PROVIDER) == "openai":
            # Chat mode
            message = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": intro}],
                }
            ]
            for x, y, z in examples:
                example_img = Image.open(z)
                if not self.no_img:
                    content = [
                        {"type": "text", "text": x},
                        {
                            "type": "text",
                            "text": "IMAGES: (1) desktop screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(example_img)},
                        },
                    ]
                else:
                    content = [
                        {"type": "text", "text": x},
                    ]

                message.append(
                    {"role": "system", "name": "example_user", "content": content}
                )
                message.append(
                    {
                        "role": "system",
                        "name": "example_assistant",
                        "content": [{"type": "text", "text": y}],
                    }
                )

            # Encode images and page_screenshot_img as base64 strings.
            current_prompt = current
            if not self.no_img:
                content = [
                    {
                        "type": "text",
                        "text": "IMAGES: (1) desktop screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(page_screenshot_img)},
                    },
                ]
            else:
                content = []

            for image_i, image in enumerate(images):
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": f"({image_i+2}) input image {image_i+1}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(image)},
                        },
                    ]
                )
            content = [{"type": "text", "text": current_prompt}] + content

            message.append({"role": "user", "content": content})
            return message
        elif get_config(ConfigKey.PROVIDER) == "anthropic":
            # Chat mode
            message = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": intro}],
                }
            ]
            for x, y, z in examples:
                example_img = Image.open(z)
                if not self.no_img:
                    content = [
                        {"type": "text", "text": x},
                        {
                            "type": "text",
                            "text": "IMAGES: (1) desktop screenshot",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": pil_to_b64(example_img),
                            },
                        },
                    ]
                else:
                    content = [
                        {"type": "text", "text": x},
                    ]

                message.append(
                    {"role": "system", "name": "example_user", "content": content}
                )
                message.append(
                    {
                        "role": "system",
                        "name": "example_assistant",
                        "content": [{"type": "text", "text": y}],
                    }
                )

            # Encode images and page_screenshot_img as base64 strings.
            current_prompt = current
            if not self.no_img:
                content = [
                    {
                        "type": "text",
                        "text": "IMAGES: (1) desktop screenshot",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": pil_to_b64(page_screenshot_img),
                        },
                    },
                ]
            else:
                content = []

            content = [{"type": "text", "text": current_prompt}] + content

            message.append({"role": "user", "content": content})
            return message
        elif (
            get_config(ConfigKey.PROVIDER) == "groq"
            or get_config(ConfigKey.PROVIDER) == "ollama"
        ):
            if not self.no_img:
                raise Exception("Current ollama and groq models are not multimodal")
            # Chat mode
            message = [
                {
                    "role": "system",
                    "content": intro,
                }
            ]
            for x, y, z in examples:
                message.append({"role": "system", "name": "example_user", "content": x})
                message.append(
                    {
                        "role": "system",
                        "name": "example_assistant",
                        "content": y,
                    }
                )

            # Encode images and page_screenshot_img as base64 strings.
            message.append({"role": "user", "content": current})
            return message
        elif get_config(ConfigKey.PROVIDER) == "google":
            if "chat" in get_config(ConfigKey.PROMPT_MODE):
                message = [
                    intro,
                    "Here are a few examples:",
                ]
                for x, y, z in examples:
                    example_img = Image.open(z)
                    message.append(f"Observation\n:{x}\n")
                    if not self.no_img:
                        message.extend(
                            [
                                "IMAGES:",
                                "(1) desktop screenshot:",
                                pil_to_vertex(example_img),
                            ]
                        )
                    else:
                        message.extend(["IMAGES:"])
                    message.append(f"Action: {y}")
                message.append(
                    "Those were the examples. Now make a prediction given the observation"
                )
                message.append(f"{current}\n")
                if not self.no_img:
                    message.extend(
                        [
                            "IMAGES:",
                            "(1) desktop screenshot:",
                            pil_to_vertex(page_screenshot_img),
                        ]
                    )
                else:
                    message.extend(["IMAGES:"])
                for image_i, image in enumerate(images):
                    message.extend(
                        [
                            f"({image_i+2}) input image {image_i+1}",
                            pil_to_vertex(image),
                        ]
                    )
                message.append("Let's think step by step.\n")
                return message
            elif "single" in get_config(ConfigKey.PROMPT_MODE):
                message = intro + "\nHere are a few examples:\n"
                message_images = []
                image_idx = 1
                for x, y, z in examples:
                    example_img = Image.open(z)
                    message += f"Example {image_idx}:\n\nObservation\n:{x}\n"
                    if not self.no_img:
                        message_images.append(pil_to_vertex(example_img))
                        # message_images.append(example_img.convert("RGB"))
                        message += "IMAGES:\ndesktop screenshot ({})\n".format(
                            image_idx
                        )
                    message += f"Action: {y}\n"
                    image_idx += 1
                message += "Those were the examples. Now make a prediction given the observation\n"
                message += f"{current}\n"
                if not self.no_img:
                    message_images.append(pil_to_vertex(page_screenshot_img))
                    # message_images.append(page_screenshot_img.convert("RGB"))
                    message += "IMAGES:\ndesktop screenshot ({})\n".format(image_idx)

                message += "Let's think step by step.\n"

                message = [message]
                message.extend(message_images)
            else:
                raise NotImplementedError(
                    "Prompt mode {} not implemented".format(
                        get_config(ConfigKey.PROMPT_MODE)
                    )
                )

            return message
        else:
            raise NotImplementedError(
                "Provider {} not implemented".format(get_config(ConfigKey.MODEL_NAME))
            )
