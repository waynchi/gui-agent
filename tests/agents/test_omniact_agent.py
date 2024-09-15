import pytest
from unittest.mock import Mock, MagicMock
from agents.omniact_agent import OmniactAgent


@pytest.fixture
def agent():
    # Create a mock logger
    logger = Mock()
    # Mock other dependencies as needed
    model_name = "model"
    response_parser = Mock()
    data_saver = Mock()

    # Instantiate your agent with the mocked dependencies
    agent = OmniactAgent(model_name, response_parser, logger, data_saver)
    return agent


def test_parse_command_click(agent):
    id2center = {"1234": (640, 480)}
    command = "click [1234]"
    expected = "pyautogui.click(640, 480)"
    assert agent.parse_command(command, id2center) == expected


def test_parse_command_invalid_id(agent):
    id2center = {"1234": (640, 480)}
    command = "click [5678]"
    expected = ""
    assert agent.parse_command(command, id2center) == expected
    agent.logger.error.assert_called_with("# ID 5678 not found in id2center")


def test_parse_command_unsupported_action(agent):
    id2center = {"1234": (640, 480)}
    command = "unsupported_action [1234]"
    expected = ""
    assert agent.parse_command(command, id2center) == expected
    agent.logger.error.assert_called_with("Unsupported action: unsupported_action")


def test_response_to_pyautogui_click_and_type(agent):
    id2center = {"1234": (640, 480)}

    response = (
        "In summary, the actions I will perform are ```type [1234] [Hello World] [1]```"
    )
    expected = 'pyautogui.write("Hello World")\npyautogui.press("enter")'
    assert agent.response_to_pyautogui(response, id2center) == expected

    response = "In summary, the actions I will perform are ```click [1234]\ntype [1234] [Hello World] [0]```"
    expected = 'pyautogui.click(640, 480)\npyautogui.write("Hello World")'
    assert agent.response_to_pyautogui(response, id2center) == expected


def test_response_to_pyautogui_press_and_hotkey(agent):
    id2center = {"1234": (640, 480)}

    response = "In summary, the actions I will perform are ```press [space]```"
    expected = 'pyautogui.press("space")'
    assert agent.response_to_pyautogui(response, id2center) == expected

    response = (
        "In summary, the actions I will perform are ```hotkey [ctrl] [alt] [delete]```"
    )
    expected = "pyautogui.hotkey(ctrl, alt, delete)"
    assert agent.response_to_pyautogui(response, id2center) == expected


def test_response_to_pyautogui_with_invalid_id(agent):
    id2center = {"1234": (640, 480)}
    response = "In summary, the actions I will perform are ```click [5678]```"
    expected = ""
    assert agent.response_to_pyautogui(response, id2center) == expected


# More tests can be added for other scenarios and combinations of actions
