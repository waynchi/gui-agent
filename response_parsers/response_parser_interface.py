from actions.action import Action
from abc import ABC, abstractmethod


class ResponseParserInterface(ABC):
    @abstractmethod
    def response_to_action(self, response, id2center=None) -> Action:
        """
        Parses the response from the LLM Agent and converts into an action
        """
        pass
