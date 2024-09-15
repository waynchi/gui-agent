from actions.action import Action
from response_parsers.response_parser_interface import ResponseParserInterface
from environments.miniwob_environment import MiniWobEnvironment


class MiniWobSoMResponseParser(ResponseParserInterface):
    def __init__(self, environment):
        self.environment = environment
        assert type(self.environment) == MiniWobEnvironment

    def response_to_action(self, response, id2center) -> Action:
        if "CLICK" in response:
            action_type = "CLICK_COORDS"
            coords = id2center[response.split(" ")[1]]
            action_params = {"coords": coords[:2]}
        elif "TYPE_TEXT" in response:
            action_type = "TYPE_TEXT"
            text = " ".join(response.split(" ")[1:])
            action_params = {"text": text}

        return Action(action_type, action_params, "None", response)
