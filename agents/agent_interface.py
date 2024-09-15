from environments.task import Task
from typing import List
from prompt_generators.prompt import Prompt
from response_parsers.response_parser_interface import ResponseParserInterface
from actions.action import Action
from utils.data_saver import DataSaver
from abc import ABC, abstractmethod


class AgentInterface(ABC):
    def __init__(
        self,
        model_name: str,
        response_parser: ResponseParserInterface,
        logger,
        data_saver: DataSaver = None,
    ):
        self.model_name = model_name
        self.logger = logger
        self.response_parser = response_parser
        self.data_saver = data_saver
        self.prev_actions = []
        self.max_prev_actions = 1000

    @abstractmethod
    def predict(self, prompts: List[Prompt], state, task: Task):
        """
        Returns an action given a state and action space (can change each step)
        """
        pass

    @abstractmethod
    def get_response(self, prompts: List[Prompt]):
        """
        Returns a response given a prompt
        """
        pass

    def response_to_action(self, response) -> Action:
        try:
            self.logger.debug("Response: {}".format(response))
            if self.data_saver is not None:
                self.data_saver.save_response(response)
            action = self.response_parser.response_to_action(response)
        except Exception as e:
            self.logger.error("Error parsing response: {}".format(response))
            raise e
        return action
