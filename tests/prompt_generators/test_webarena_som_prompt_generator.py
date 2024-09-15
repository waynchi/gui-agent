from environments.task import Task
from unittest.mock import MagicMock
from collections import Counter as count
from PIL import Image as PILImage
from typing import Any
import pytest
from prompt_generators.webarena_som_prompt_generator import (
    WebArenaSoMPromptGenerator,
    SoMState,
)
from prompt_generators.prompt import PromptRoleType
from environments.webarena_environment import WebArenaEnvironment
from utils.config_parser import load_yaml


@pytest.fixture(autouse=True)
def load_config():
    load_yaml("tests/vwa_config.yaml")


class MockedEnv(MagicMock, WebArenaEnvironment):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)


@pytest.fixture
def generator():
    # Fixture to create a WebArenaSoMPromptGenerator instance
    environment = MockedEnv()
    logger = MagicMock()
    return WebArenaSoMPromptGenerator(environment, logger)


def test_construct_example_prompt(generator):
    example = {
        "text": "Example text",
        "response": "Example response",
        "image": "prompt_generators/example_prompts/som_examples/som_example1.png",
    }
    prompts = generator.construct_example_prompt(example)
    assert len(prompts) == 2
    assert prompts[0].text == "Example text"


def test_generate_prompts(generator):
    state = SoMState(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    task = Task(
        "Example task",
        PILImage.new("RGB", (1280, 720), (255, 255, 255)),
    )
    prev_actions = ["Previous action 1", "Previous action 2"]

    prompts = generator.generate_prompts(state, task, prev_actions)
    assert len(prompts) == 8
    assert any(prompt.role == PromptRoleType.SYSTEM for prompt in prompts)
    assert any(prompt.role == PromptRoleType.USER for prompt in prompts)
    assert all(prompt.role == PromptRoleType.SYSTEM for prompt in prompts[0:7])
    assert prompts[-1].role == PromptRoleType.USER
