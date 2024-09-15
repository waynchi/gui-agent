from typing import List
from PIL.Image import Image
from dataclasses import dataclass


@dataclass
class Task:
    task: str
    images: List[Image]
