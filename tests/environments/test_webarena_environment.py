import pytest
from PIL import Image as PILImage
from unittest.mock import MagicMock
from environments.webarena_environment import WebArenaEnvironment
from typing import Any
from prompt_generators.webarena_som_prompt_generator import WebArenaSoMPromptGenerator
from actions.action import Action
from browser_env import (
    create_mouse_click_action,
    create_keyboard_type_action,
    create_key_press_action,
    create_scroll_action,
    create_stop_action,
)


class MockedEnv(MagicMock, WebArenaEnvironment):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)

    def get_viewport_size(self):
        return {"height": 720, "width": 1280}


@pytest.fixture(autouse=True)
def environment():

    env = MockedEnv(
        "vwa_config_files/test_reddit/0.json",
        MagicMock(),
        MagicMock(),
        render_mode=None,
    )

    return env


def test_open_image_from_url(environment):
    url = "https://picsum.photos/200/300"
    image = environment.open_image_from_url(url)
    assert image is not None
    assert isinstance(image, PILImage.Image)


def test_action_to_web_arena_click_coords(environment):
    viewport_size = environment.get_viewport_size()
    action = Action(
        "CLICK_COORDS",
        {"coords": (viewport_size["width"] / 2.0, viewport_size["height"] / 2.0)},
        "None",
        "",
    )
    expected_action = [create_mouse_click_action(0.5, 0.5)]
    result = environment.action_to_web_arena_action(action)
    assert len(result) == 1
    assert result[0]["action_type"] == expected_action[0]["action_type"]


def test_action_to_web_arena_type_text(environment):
    action = Action("TYPE_TEXT", {"text": "hello", "press_enter": True}, "None", "")
    expected_actions = [
        create_keyboard_type_action("hello"),
        create_key_press_action("Enter"),
    ]
    result = environment.action_to_web_arena_action(action)
    assert len(result) == 2
    assert result[0]["action_type"] == expected_actions[0]["action_type"]
    assert result[1]["action_type"] == expected_actions[1]["action_type"]


def test_action_to_web_arena_scroll(environment):
    action = Action("SCROLL", {"direction": "down"}, "None", "")
    expected_action = [create_scroll_action("down")]
    result = environment.action_to_web_arena_action(action)
    assert len(result) == 1
    assert result[0]["action_type"] == expected_action[0]["action_type"]


def test_action_to_web_arena_stop(environment):
    action = Action("STOP", {"answer": "stop"}, "None", "")
    expected_action = [create_stop_action("stop")]
    result = environment.action_to_web_arena_action(action)
    assert len(result) == 1
    assert result[0]["action_type"] == expected_action[0]["action_type"]


def test_action_to_web_arena_invalid(environment):
    action = Action("INVALID", {}, "None", "")
    with pytest.raises(Exception):
        environment.action_to_web_arena_action(action)
