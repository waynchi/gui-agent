from actions.action import Action


class TestAction(object):
    def test_to_json(self):
        action = Action(
            action_type="test",
            action_params={"param1": "value1", "param2": "value2"},
            description="Test action",
            response="Test response",
        )
        expected_json = {
            "action": "test",
            "params": {"param1": "value1", "param2": "value2"},
            "description": "Test action",
            "response": "Test response",
        }
        assert action.to_json() == expected_json

    def test_eq(self):
        action1 = Action(
            action_type="test",
            action_params={"param1": "value1", "param2": [1, 2]},
            description="Test action",
            response="Test response",
        )
        action2 = Action(
            action_type="test",
            action_params={"param1": "value1", "param2": [1, 2]},
            description="Test action",
            response="Test response",
        )
        action3 = Action(
            action_type="different",
            action_params={"param1": "value1", "param2": [1, 2]},
            description="Test action",
            response="Test response",
        )
        action4 = Action(
            action_type="test",
            action_params={"param1": "value1", "param2": [1, 2, 3]},
            description="Test action",
            response="Test response",
        )
        assert action1 == action2
        assert action1 != action3
        assert action1 != action4

    def test_hash(self):
        action1 = Action(
            action_type="test",
            action_params={"param1": "value1", "param2": [1, 2]},
            description="Test action",
            response="Test response",
        )
        action2 = Action(
            action_type="test",
            action_params={"param1": "value1", "param2": [1, 2]},
            description="Test action",
            response="Test response",
        )
        action3 = Action(
            action_type="different",
            action_params={"param1": "value1", "param2": [1, 2]},
            description="Test action",
            response="Test response",
        )
        action4 = Action(
            action_type="test",
            action_params={"param1": "value1", "param2": [1, 2, 3]},
            description="Test action",
            response="Test response",
        )
        assert hash(action1) == hash(action2)
        assert hash(action1) != hash(action3)
        assert hash(action1) != hash(action4)
