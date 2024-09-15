import re
from response_parsers.response_parser_interface import ResponseParserInterface
from environments.webarena_environment import WebArenaEnvironment
from actions.action import Action


class WebArenaSoMResponseParser(ResponseParserInterface):
    def __init__(self, environment):
        self.environment = environment
        # assert issubclass(type(self.environment), WebArenaEnvironment)

    def extract_text_within_brackets(self, text):
        pattern = r"\[([^]]+)\]"
        matches = re.findall(pattern, text)
        if len(matches) == 0:
            matches = ["None"]
        return matches

    def response_to_action(self, response, id2center, no_filter=False) -> Action:

        # Get text inside ``` ```
        if no_filter:
            filtered_response = response
        else:
            filtered_response = response.split("```")
            if len(filtered_response) <= 1:
                return Action("NONE", {}, "None", response)

            filtered_response = filtered_response[-2]
            # print(filtered_response)

        if "CLICK" in filtered_response:
            # TODO Enumerate
            action_type = "CLICK_COORDS"
            coords = id2center[self.extract_text_within_brackets(filtered_response)[-1]]
            action_params = {"coords": coords[:2]}
        elif "TYPE_TEXT" in filtered_response:
            action_type = "TYPE_TEXT"
            text = self.extract_text_within_brackets(filtered_response)[-2]
            press_enter = (
                self.extract_text_within_brackets(filtered_response)[-1] == "1"
            )
            coords = id2center[self.extract_text_within_brackets(filtered_response)[-3]]
            action_params = {
                "text": text,
                "press_enter": press_enter,
                "coords": coords[:2],
            }
        elif "SCROLL" in filtered_response:
            action_type = "SCROLL"
            text = self.extract_text_within_brackets(filtered_response)[-1]
            action_params = {"direction": text}
        elif "STOP" in filtered_response:
            action_type = "STOP"
            text = self.extract_text_within_brackets(filtered_response)[-1]
            action_params = {"answer": text}
        else:
            raise ValueError(f"Unknown action type: {filtered_response} in {response}")

        return Action(action_type, action_params, "None", response)
