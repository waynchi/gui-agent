from typing import TypedDict, List


class RunResult(TypedDict):
    reward: int
    error: bool
    error_string: str
    actions: List[dict]
