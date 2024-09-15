from environments.task import Task
from PIL import Image
from prompt_generators.example_prompts.webarena_som_cot_prompt import prompt_info
from prompt_generators.prompt import Prompt, PromptRoleType, CaptionedImage
from typing import List
import json
import numpy as np
from prompt_generators.prompt_generator_interface import (
    PromptGeneratorInterface,
)
from environments.webarena_environment import WebArenaEnvironment
from transformers import GPT2Tokenizer
from transformers.utils import logging
from utils.config_parser import get_config, ConfigKey
from environments.set_of_mark import SoMState

logging.set_verbosity(40)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class WebArenaSoMPromptGenerator(PromptGeneratorInterface):
    """
    Mini Wob Set of Marks prompt generator
    """

    def __init__(self, environment, logger):
        super().__init__(environment, logger)
        assert issubclass(type(self.environment), WebArenaEnvironment)
        # Since we don't have access to GPT-3's tokenizer, we use GPT-2 instead
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def construct_example_prompt(self, example):
        return [
            Prompt(
                text=example["text"],
                captioned_images=[
                    CaptionedImage(
                        image=Image.open(example["image"]),
                        caption="IMAGES: (1) Current Page Screenshot",
                    )
                ],
                role=PromptRoleType.SYSTEM,
                name="example_user",
                tags=[],
            ),
            Prompt(
                text=example["response"],
                captioned_images=[],
                role=PromptRoleType.SYSTEM,
                name="example_assistant",
                tags=[],
            ),
        ]

    def generate_prompts(
        self, state: SoMState, task: Task, prev_actions: List[str], last_step=False
    ) -> List[Prompt]:
        """
        Returns a prompt given a state and action space (can change each step)
        """

        intro_prompt = Prompt(
            text=prompt_info["intro"],
            captioned_images=[],
            role=PromptRoleType.SYSTEM,
            name=None,
            tags=[],
        )

        prompts = [intro_prompt]

        for example in prompt_info["examples"]:
            prompts.extend(self.construct_example_prompt(example))

        som_captions = self.truncate_to_token_limit(
            state.acc_tree, self.tokenizer, 3840
        )
        user_prompt_text = prompt_info["template"].format(
            state.url, som_captions, task.task, "\n".join(prev_actions)
        )
        # user_prompt_text = self.truncate_to_token_limit(
        #     user_prompt_text, self.tokenizer, get_config(ConfigKey.TOKEN_LIMIT)
        # )
        if last_step:
            user_prompt_text += "This is the last step in the plan. You must output STOP and respond with an answer.\n"

        captioned_images = [
            CaptionedImage(
                image=state.som_image,
                caption="IMAGES: (1) Current Page Screenshot",
            )
        ]

        if task.images is not None:
            for i in range(len(task.images)):
                captioned_images.append(
                    CaptionedImage(
                        image=task.images[i],
                        caption="IMAGES: ({}) Requested Image".format(i + 2),
                    )
                )

        prompts.append(
            Prompt(
                text=user_prompt_text,
                captioned_images=captioned_images,
                role=PromptRoleType.USER,
                name=None,
                tags=[],
            )
        )

        return prompts
