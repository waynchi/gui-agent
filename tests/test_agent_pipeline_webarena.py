from typing import Any
import pytest
from unittest.mock import MagicMock, patch
from utils.config_parser import ConfigKey, load_yaml
from actions.action import Action
from environments.set_of_mark import SoMState

from pipeline.agent_pipeline import AgentPipeline
from unittest.mock import MagicMock, patch
from environments.webarena_environment import WebArenaEnvironment


# Load YAML configuration before every test
@pytest.fixture(autouse=True)
def load_config():
    load_yaml("tests/vwa_config.yaml")


@pytest.fixture
def logger():
    return MagicMock()


@pytest.fixture
def webarena_env_name():
    return "vwa_config_files/test_reddit/0.json"


def test_get_environment_vwa(logger, webarena_env_name):
    pipeline = AgentPipeline(logger, MagicMock())
    environment = pipeline.get_environment("vwa", webarena_env_name, render_mode=None)
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


def test_run_pipeline(logger, webarena_env_name):
    pipeline = AgentPipeline(logger, MagicMock())

    class MockedAgent(MagicMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prev_actions = [Action("NONE", {}, "", "")]

        def predict(self, prompts, state, task):
            return self.prev_actions[0]

    class MockedEnv(MagicMock, WebArenaEnvironment):
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)

        def get_state(self, *args, **kwargs):
            return SoMState(MagicMock(), MagicMock(), MagicMock(), MagicMock())

        def execute_action(self, action: Action, step_id: int):
            return 1, True

    mocked_agent = MockedAgent()
    pipeline.get_agent = mocked_agent
    pipeline.get_environment = MockedEnv()
    run_result = pipeline.run_pipeline(webarena_env_name, render_mode=None)
    assert run_result is not None
    assert run_result["error"] is False
    assert run_result["error_string"] is ""
    assert len(run_result["actions"]) == 1
    assert run_result["actions"][0] == mocked_agent.prev_actions[0].to_json()
