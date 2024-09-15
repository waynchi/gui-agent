from typing import Any
import pytest
from response_parsers.webarena_som_response_parser import WebArenaSoMResponseParser
from environments.webarena_environment import WebArenaEnvironment
from actions.action import Action
from unittest.mock import MagicMock


@pytest.fixture
def parser():
    class MockedEnv(MagicMock, WebArenaEnvironment):
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)

    environment = MockedEnv()
    return WebArenaSoMResponseParser(environment)


def test_extract_text_within_brackets(parser):
    test_str = "This is a test [extract this] string."
    result = parser.extract_text_within_brackets(test_str)
    assert result == ["extract this"]


def test_extract_text_within_brackets_empty(parser):
    test_str = "STOP []"
    result = parser.extract_text_within_brackets(test_str)
    assert result == ["None"]


def test_response_to_action_click(parser):
    response = "```Some text CLICK [1]```"
    id2center = {"1": (100, 200)}
    expected_action = Action("CLICK_COORDS", {"coords": (100, 200)}, "None", response)
    action = parser.response_to_action(response, id2center)
    assert action == expected_action


def test_response_to_action_type_text(parser):
    response = "```Some text TYPE_TEXT [1] [text] [1]```"
    id2center = {"1": (100, 200)}
    expected_action = Action(
        "TYPE_TEXT",
        {"text": "text", "press_enter": True, "coords": (100, 200)},
        "None",
        response,
    )
    action = parser.response_to_action(response, id2center)
    assert action == expected_action


def test_response_to_action_scroll(parser):
    response = "```Some text SCROLL [down]```"
    id2center = {}
    expected_action = Action("SCROLL", {"direction": "down"}, "None", response)
    action = parser.response_to_action(response, id2center)
    assert action == expected_action


def test_response_to_action_stop(parser):
    response = "```Some text STOP [answer]```"
    id2center = {}
    expected_action = Action("STOP", {"answer": "answer"}, "None", response)
    action = parser.response_to_action(response, id2center)
    assert action == expected_action


def test_response_to_action_invalid(parser):
    response = "Invalid action"
    id2center = {}
    expected_action = Action("NONE", {}, "None", response)
    action = parser.response_to_action(response, id2center)
    assert action == expected_action


def test_failure_case(parser):
    response = "I'm sorry, but I am not able to assist with requests involving images that require me to perform a visual analysis to count or identify specific objects. My capabilities are limited to providing information, answering questions, and performing web-based tasks that do not involve interpreting images in this manner. If you have any other questions or need assistance with a different task, feel free to ask!"
    id2center = {}
    expected_action = Action("NONE", {}, "None", response)
    action = parser.response_to_action(response, id2center)
    assert action == expected_action
