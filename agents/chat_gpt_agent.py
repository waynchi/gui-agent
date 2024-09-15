from typing import List
from prompt_generators.prompt import Prompt
import time
from agents.agent_interface import AgentInterface
from response_parsers.response_parser_interface import ResponseParserInterface
from actions.action import Action
from utils.openai_utils import create_client
from utils.config_parser import get_config, ConfigKey, load_yaml
import openai


class GPTChatAgent(AgentInterface):
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

    def get_response(self, prompts: List[Prompt]):
        retries = 0
        while retries < get_config(ConfigKey.RATE_LIMIT_MAX_RETRIES):
            try:
                messages = [prompt.construct_chat_gpt_message() for prompt in prompts]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1000,
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

    def predict(self, prompts: List[Prompt], state, task):
        """
        Returns an action given a state and action space (can change each step)
        """
        retries = 0
        action = None
        while retries < get_config("agent_predict_max_retries"):
            try:
                response = self.get_response(prompts)
                action = self.response_to_action(response)
                break
            except Exception as e:
                self.logger.error("Retrying. Error getting response: {}".format(e))
                retries += 1

        if action is None:
            raise Exception("Exceeded Maximum Retries for Getting a Response (Predict)")

        self.prev_actions.append(action)
        if len(self.prev_actions) > self.max_prev_actions:
            self.prev_actions = self.prev_actions[-self.max_prev_actions :]

        return action
