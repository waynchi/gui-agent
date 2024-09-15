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


class MiniWobPromptGenerator(PromptGeneratorInterface):
    def __init__(self, environment, logger):
        super().__init__(environment, logger)
        assert type(self.environment) == MiniWobEnvironment
        # Since we don't have access to GPT-3's tokenizer, we use GPT-2 instead
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def generate_prompts(self, state, task, prev_actions, last_step=False):
        """
        Returns a prompt given a state and action space (can change each step)
        """

        task_prompt = task
        state_prompt = {"dom": state["dom_elements"], "fields": state["fields"]}

        examples = [
            {
                "description": "Click on the element with reference 5",
                "action": "CLICK_ELEMENT",
                "params": {"ref": 5},
            },
            # {
            #     "description": "Click the button at x=35, y=50",
            #     "action": "CLICK_COORDS",
            #     "params": {
            #         "coords": [35, 50]
            #     },
            # },
            {
                "description": "Type the text 'hello world'",
                "action": "TYPE_TEXT",
                "params": {"text": "hello world"},
            },
            {
                "description": "Click on the element with reference 5 and type text 'hello world'",
                "action": "FOCUS_ELEMENT_AND_TYPE_TEXT",
                "params": {"ref": 5, "text": "hello world"},
            },
            # {
            #     "description": "Scroll up at coordinate x=50, y=60",
            #     "action": "SCROLL_UP_COORDS",
            #     "params": {
            #         "coords": [55, 60]
            #     },
            # },
        ]

        output = {
            "description": "Describe the single action here",
            "action": "ACTION_TYPE",
            "params": {"ACTION_PARAM": "VALUE"},
        }

        prompt = "Complete the following task: {} \n\n".format(task_prompt)
        prompt += "This is the current state of the webpage: \n {} \n\n".format(
            json.dumps(state_prompt, cls=NumpyEncoder)
        )
        prompt += "These are some examples of actions: \n {} \n\n".format(
            json.dumps(examples)
        )
        prompt += "These are the previous actions: \n {} \n\n".format(
            json.dumps([prev_action.to_json() for prev_action in prev_actions])
        )

        prompt = self.truncate_to_token_limit(
            prompt, self.tokenizer, get_config(ConfigKey.TOKEN_LIMIT)
        )

        # prompt += "Output a sequence or plan of actions in the following format: \n {} \n\n".format(json.dumps(output, cls=NumpyEncoder))

        prompt += "Output a single action in the following format without any other text: \n {}".format(
            json.dumps(output)
        )

        return prompt
