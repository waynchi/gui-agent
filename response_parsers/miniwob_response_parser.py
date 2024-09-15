import ast
import json
from actions.action import Action
from response_parsers.response_parser_interface import ResponseParserInterface
from environments.miniwob_environment import MiniWobEnvironment


class MiniWobResponseParser(ResponseParserInterface):
    def __init__(self, environment):
        self.environment = environment
        assert type(self.environment) == MiniWobEnvironment

    def response_to_action(self, response):
        response = json.loads(response)
        description = response["description"]
        action_type = response["action"]
        action_params = response["params"]
        if "coords" in action_params.keys():
            try:
                action_params["coords"] = ast.literal_eval(action_params["coords"])
            except:
                pass
        return Action(action_type, action_params, description, response)
