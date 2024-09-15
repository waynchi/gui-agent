from typing import List
from environments.set_of_mark import SoMState
from abc import ABC, abstractmethod
from prompt_generators.prompt import Prompt
from utils.config_parser import get_config, ConfigKey


class PromptGeneratorInterface(ABC):
    def __init__(self, environment, logger):
        """
        Initializes the prompt generator
        """
        self.environment = environment
        self.logger = logger

    @abstractmethod
    def generate_prompts(
        self, state: SoMState, task: str, prev_actions: List[str], last_step=False
    ) -> List[Prompt]:
        """
        Returns a prompt for the environment
        """
        pass

    def truncate_to_token_limit(self, text_prompt, tokenizer, max_tokens):
        """Truncate the prompt to the maximum token limit"""
        self.logger.info("Obs char count: {}".format(len(text_prompt)))
        if get_config(ConfigKey.PROVIDER) == "openai":
            tokens = tokenizer.encode(text_prompt)

            self.logger.info("Obs token count: {}".format(len(tokens)))
            if len(tokens) > max_tokens:
                self.logger.info("Truncating prompt to {} tokens".format(max_tokens))
                truncated_tokens = tokens[:max_tokens]
                return tokenizer.decode(truncated_tokens)
        elif get_config(ConfigKey.PROVIDER) == "google":
            if len(text_prompt) > max_tokens:
                self.logger.info(
                    "Gemini model uses characters. Truncating prompt to {} characters".format(
                        max_tokens
                    )
                )
                return text_prompt[:max_tokens]
        elif (
            get_config(ConfigKey.PROVIDER) == "groq"
            or get_config(ConfigKey.PROVIDER) == "ollama"
        ):
            tokens = tokenizer.encode(text_prompt)

            self.logger.info("Obs token count: {}".format(len(tokens)))
            if len(tokens) > max_tokens:
                self.logger.info("Truncating prompt to {} tokens".format(max_tokens))
                truncated_tokens = tokens[:max_tokens]
                return tokenizer.decode(truncated_tokens)
        else:
            raise NotImplementedError("Provider not supported")

        return text_prompt
