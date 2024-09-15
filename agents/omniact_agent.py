import re
import os
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
from PIL.Image import Image
from utils.time_utils import time_function
import ollama
from groq import Groq


from google.api_core.exceptions import InvalidArgument
from vertexai.preview.generative_models import (
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
)
import google.generativeai as genai


class OmniactAgent(AgentInterface):
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
        if get_config(ConfigKey.PROVIDER) == "openai":
            self.client = create_client()
        elif get_config(ConfigKey.PROVIDER) == "google":
            self.google_model = GenerativeModel(get_config(ConfigKey.MODEL_NAME))
        elif get_config(ConfigKey.PROVIDER) == "groq":
            if not get_config(ConfigKey.NO_IMG):
                # Might not be true, but true for our use case (llama-3)
                raise Exception("GROQ does not support image input")
            self.groq_client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )

    @time_function
    def get_response(self, prompts: List[str]):
        retries = 0
        while retries < get_config(ConfigKey.RATE_LIMIT_MAX_RETRIES):
            try:
                messages = prompts
                temperature = get_config(ConfigKey.TEMPERATURE)
                top_p = get_config(ConfigKey.TOP_P)

                if get_config(ConfigKey.PROVIDER) == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=1000,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    response = response.choices[0].message.content
                elif get_config(ConfigKey.PROVIDER) == "google":
                    safety_config = {
                        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    }
                    response = self.google_model.generate_content(
                        messages,
                        generation_config=dict(
                            candidate_count=1,
                            max_output_tokens=1000,
                            top_p=top_p,
                            temperature=temperature,
                        ),
                        safety_settings=safety_config,
                    )
                    response = response.text
                elif get_config(ConfigKey.PROVIDER) == "groq":
                    response = self.groq_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=1000,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    print(response.usage)
                    response = response.choices[0].message.content
                elif get_config(ConfigKey.PROVIDER) == "ollama":
                    response = ollama.chat(
                        model=self.model_name,
                        messages=messages,
                        options={
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_tokens": 1000,
                        },
                    )
                    response = response["message"]["content"]
                return response
            except openai.RateLimitError as e:
                # TODO While this fixes the rate limit issue,
                # for miniwob the model will fail anyways since there's a time limit on tasks
                self.logger.error("Rate limit error: {}".format(e))
                self.logger.error("Waiting 1 minute before retrying")
                time.sleep(60)
                retries += 1
            except Exception as e:
                self.logger.error(
                    "Encountered a Potential rate limit error: {}".format(e)
                )
                self.logger.error(traceback.format_exc())
                self.logger.info("WAITING for 20 seconds before retrying.")
                time.sleep(20)
                retries += 1
        raise Exception("Exceeded Maximum Retries for Getting a Response (Rate Limit)")

    def parse_command(self, command, id2center):
        # Split the command by spaces and brackets
        parts = command.split()
        action = parts[0]  # The first part is the action
        args = command[len(action) :].strip()  # Remaining part of the command

        # Extract arguments from brackets
        args_list = []
        while "[" in args:
            start = args.index("[") + 1
            end = args.index("]")
            args_list.append(args[start:end])
            args = args[end + 1 :].strip()

        # Handle the action by mapping to PyAutoGUI functions
        if action in ["click", "double_click", "right_click", "hover"]:
            if args_list[0] not in id2center:
                self.logger.error(f"# ID {args_list[0]} not found in id2center")
                return ""
            coords = id2center[args_list[0]]
            if action == "hover":
                return f"pyautogui.moveTo({coords[0]}, {coords[1]})"
            elif action == "double_click":
                return f"pyautogui.click({coords[0]}, {coords[1]})\npyautogui.click({coords[0]}, {coords[1]})"
            elif action == "right_click":
                return f"pyautogui.rightClick({coords[0]}, {coords[1]})"
            elif action == "click":
                return f"pyautogui.click({coords[0]}, {coords[1]})"
        elif action == "type":
            # if args_list[0] not in id2center:
            #     self.logger.error(f"# ID {args_list[0]} not found in id2center")
            #     return ""
            # coords = id2center[args_list[0]]

            # content = args_list[1]
            # press_enter_after = args_list[2] if len(args_list) > 2 else "1"
            # if press_enter_after == "0":
            #     return f'pyautogui.click({coords[0]}, {coords[1]})\npyautogui.write("{content}")'
            # else:
            #     return f'pyautogui.click({coords[0]}, {coords[1]})\npyautogui.write("{content}")\npyautogui.press("enter")'
            content = args_list[0]
            return f'pyautogui.write("{content}")'
            # press_enter_after = args_list[1] if len(args_list) > 1 else "1"
            # if press_enter_after == "0":
            #     return f'pyautogui.write("{content}")'
            # else:
            #     return f'pyautogui.write("{content}")\npyautogui.press("enter")'
        elif action == "press":
            return f'pyautogui.press("{args_list[0]}")'
        elif action == "hotkey":
            hotkeys = ", ".join(args_list)
            return "pyautogui.hotkey({})".format(hotkeys)
        else:
            self.logger.error(f"Unsupported action: {action}")
            return ""

    @time_function
    def response_to_pyautogui(self, response: str, id2center) -> str:
        action_splitter = "```"
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)

        pyautogui_commands = []
        commands = match.group(1).strip().split("\n")
        if get_config(ConfigKey.ACTION_TYPES) == "omniact_pyautogui":
            pyautogui_commands = commands
        elif get_config(ConfigKey.ACTION_TYPES) == "omniact_actions":
            for command in commands:
                try:
                    pyautogui_command = self.parse_command(command, id2center)
                    pyautogui_commands.append(pyautogui_command)
                except:
                    self.logger.error(
                        "Skipping command. Unable to parse command: {}".format(command)
                    )

        return "\n".join(pyautogui_commands)

    def predict(self, prompts: List[Prompt], state: SoMState, task):
        """
        Returns an action given a state and action space (can change each step)
        """
        try:
            response = self.get_response(prompts)
            pyautogui_commands = self.response_to_pyautogui(response, state.id2center)
            self.logger.debug("response: {}".format(response))

            return pyautogui_commands, response
        except Exception as e:
            self.logger.error(
                "Retrying. Error getting or parsing response: {}".format(e)
            )
            self.logger.error(traceback.format_exc())
            self.logger.error("response: {}".format(response))
            raise Exception("Error getting or parsing response")
            # return "", response
