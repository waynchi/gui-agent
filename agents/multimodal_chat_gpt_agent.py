import traceback
from typing import List
from prompt_generators.prompt import Prompt, PromptRoleType
import time
from agents.agent_interface import AgentInterface
from environments.environment_interface import EnvironmentInterface
from response_parsers.response_parser_interface import ResponseParserInterface
from actions.action import Action
from utils.openai_utils import create_client
from utils.config_parser import get_config, ConfigKey, load_yaml
from environments.set_of_mark import SoMState
import openai
from utils.openai_utils import encode_image
from PIL.Image import Image


class GPTSoMAgent(AgentInterface):
    """
    GPT without vision
    """

    def __init__(
        self,
        model_name: str,
        response_parser: ResponseParserInterface,
        logger,
        data_saver,
    ):
        super().__init__(model_name, response_parser, logger, data_saver=data_saver)
        self.client = create_client()
        self.prev_responses = []

    def get_response(self, prompts: List[Prompt]):
        retries = 0
        while retries < get_config(ConfigKey.RATE_LIMIT_MAX_RETRIES):
            try:
                messages = [prompt.construct_chat_gpt_message() for prompt in prompts]

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1000,
                    temperature=1.0,
                    top_p=0.9,
                )
                response = response.choices[0].message.content
                return response
            except openai.RateLimitError as e:
                # TODO While this fixes the rate limit issue,
                # for miniwob the model will fail anyways since there's a time limit on tasks
                self.logger.error("Rate limit error: {}".format(e))
                self.logger.error("Waiting 1 minute before retrying")
                time.sleep(60)
                retries += 1
        raise Exception("Exceeded Maximum Retries for Getting a Response (Rate Limit)")

    def response_to_action(self, response, id2center) -> Action:
        try:
            self.logger.debug("Response: {}".format(response))
            self.data_saver.save_response(response)
            action = self.response_parser.response_to_action(
                response, id2center=id2center
            )
        except Exception as e:
            self.logger.error("Error parsing response: {}".format(response))
            raise e
        return action

    def predict(self, prompts: List[Prompt], state: SoMState, task):
        """
        Returns an action given a state and action space (can change each step)
        """
        retries = 0
        action = None
        while retries < get_config("agent_predict_max_retries"):
            try:
                response = self.get_response(prompts)
                self.logger.debug("response: {}".format(response))
                action = self.response_to_action(response, state.id2center)
                break
            except Exception as e:
                breakpoint()
                self.logger.error(traceback.format_exc())
                self.logger.error("Retrying. Error getting response: {}".format(e))
                retries += 1
        if action is None:
            raise Exception("Exceeded Maximum Retries for Getting a Response (Predict)")

        # self.prev_responses.append(response.split("```")[-2])
        self.prev_responses.append(response)
        self.prev_actions.append(action)
        if len(self.prev_actions) > self.max_prev_actions:
            self.prev_actions = self.prev_actions[-self.max_prev_actions :]
            self.prev_responses = self.prev_responses[-self.max_prev_actions :]

        return action
