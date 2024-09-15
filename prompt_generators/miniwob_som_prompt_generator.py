from environments.set_of_mark import SoMState
from typing import List
from prompt_generators.prompt import Prompt, PromptRoleType, CaptionedImage
import json
import numpy as np
from prompt_generators.prompt_generator_interface import (
    PromptGeneratorInterface,
)
from environments.miniwob_environment import MiniWobEnvironment
from transformers import GPT2Tokenizer
from transformers.utils import logging
from utils.config_parser import get_config, ConfigKey

logging.set_verbosity(40)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class MiniWobSoMPromptGenerator(PromptGeneratorInterface):
    """
    Mini Wob Set of Marks prompt generator
    """

    def __init__(self, environment, logger):
        super().__init__(environment, logger)
        assert issubclass(type(self.environment), MiniWobEnvironment)
        # Since we don't have access to GPT-3's tokenizer, we use GPT-2 instead
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def generate_prompts(
        self, state: SoMState, task: str, prev_actions: List[str], last_step=False
    ) -> List[Prompt]:
        """
        Returns a prompt given a state and action space (can change each step)
        """

        task_prompt = task

        prompt = "Complete the following task on the webpage: {} \n\n".format(
            task_prompt
        )
        prompt += "The state of the webpage has been give to you as an image. The image has bounding boxes and ids for each interactable element. \n\n"
        prompt += "You may CLICK on an element by specifying the element's id number. You may also TYPE_TEXT by specifying the text. \n\n"
        prompt += "For example, a plan to click button 5, click text field 3, and then type 'hello world' into field 3 will look like this: \n"
        prompt += "CLICK 5 \n"
        prompt += "CLICK 3 \n"
        prompt += "TYPE_TEXT hello world \n\n"
        prompt += "You have been executing a plan. These are the previous actions in the plan: \n "
        prompt += "\n".join(prev_actions)

        prompt = self.truncate_to_token_limit(
            prompt, self.tokenizer, get_config(ConfigKey.TOKEN_LIMIT)
        )
        prompt += "\n\n"

        prompt += "What is the next action? Output the single next action in the plan without any other text: "

        return [
            Prompt(
                text=prompt,
                captioned_images=[
                    CaptionedImage(
                        image=state["image"],
                        caption="IMAGES: (1) Current Page Screenshot",
                    )
                ],
                role=PromptRoleType.USER,
                name="user",
                tags=[],
            )
        ]
