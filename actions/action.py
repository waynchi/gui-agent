from dataclasses import dataclass
from typing import List, Any, Dict


@dataclass
class Action:
    action_type: str
    action_params: Dict[str, List[Any]]
    description: str
    response: str

    def to_json(self) -> dict:
        return {
            "action": self.action_type,
            "params": self.action_params,
            "description": self.description,
            "response": self.response,
        }

    def __eq__(self, other) -> bool:
        return (
            self.action_type == other.action_type
            and self.action_params == other.action_params
        )

    def __hash__(self) -> int:
        hashable_params = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in self.action_params.items()
        }
        return hash((self.action_type, tuple(hashable_params.items())))
