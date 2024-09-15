class ActionSpace:
    def __init__(self, action_types, config_elements, action_params):
        self.action_types = action_types
        self.action_params = action_params
        self.config_elements = config_elements

    def to_json(self):
        return {
            "action_types": [action_type.__dict__ for action_type in self.action_types],
            "config": [
                config_element.__dict__ for config_element in self.config_elements
            ],
            "action_params": [
                action_param.__dict__ for action_param in self.action_params
            ],
        }
