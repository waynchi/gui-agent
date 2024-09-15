import pytest
from unittest.mock import MagicMock, patch
from utils.config_parser import ConfigKey, load_yaml
from actions.action import Action

from pipeline.agent_pipeline import AgentPipeline
from unittest.mock import MagicMock, patch


# Load YAML configuration before every test
@pytest.fixture(autouse=True)
def load_config():
    load_yaml("tests/miniwob_config.yaml")


@pytest.fixture
def logger():
    return MagicMock()


@pytest.fixture
def miniwob_env_name():
    return "click-button"


def test_get_environment_miniwob(logger, miniwob_env_name):
    pipeline = AgentPipeline(logger, MagicMock())
    environment = pipeline.get_environment(
        "miniwob", miniwob_env_name, render_mode=None
    )
    assert environment is not None


def test_get_environment_invalid(logger):
    pipeline = AgentPipeline(logger, MagicMock())
    with pytest.raises(Exception):
        pipeline.get_environment("invalid_type", "environment_name")


def mock_get_config(key):
    if key == ConfigKey.E2E_MAX_RETRIES:
        return 1
    elif key == ConfigKey.ENVIRONMENT_TYPE:
        return "miniwob"
    elif key == ConfigKey.MAX_STEPS:
        return 1


def test_run_pipeline(logger, miniwob_env_name):
    pipeline = AgentPipeline(logger, MagicMock())

    class MockedAgent(MagicMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prev_actions = [Action("NONE", {}, "", "")]

        def predict(self, prompts, state, task):
            return self.prev_actions[0]

    mocked_agent = MockedAgent()
    pipeline.get_agent = mocked_agent
    run_result = pipeline.run_pipeline(miniwob_env_name, render_mode=None)
    assert run_result is not None
    assert run_result["error"] is False
    assert run_result["error_string"] is ""
    assert len(run_result["actions"]) == 1
    assert run_result["actions"][0] == mocked_agent.prev_actions[0].to_json()
