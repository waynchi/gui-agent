import json
from utils.openai_utils import encode_image
from PIL.Image import Image
from typing import List
from dataclasses import dataclass
from enum import Enum


class PromptRoleType(str, Enum):
    SYSTEM = "system"
    USER = "user"


@dataclass
class CaptionedImage:
    image: Image
    caption: str


@dataclass
class Prompt:
    text: str
    captioned_images: List[CaptionedImage]
    role: PromptRoleType
    name: str  # Participant name
    tags: List[str]  # Any additional information needed

    # Some commonly useful functions
    def construct_chat_gpt_message(self):
        content = []
        if self.text != "":
            content = [
                {"type": "text", "text": "{}".format(self.text)},
            ]

        for captioned_image in self.captioned_images:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": "{}".format(captioned_image.caption),
                    },
                    {
                        "type": "image_url",
                        "image_url": "data:image/png;base64,{}".format(
                            encode_image(captioned_image.image).decode("utf-8")
                        ),
                    },
                ]
            )

        message = {
            "role": self.role.value,
            "content": content,
        }

        if self.name is not None:
            message["name"] = self.name

        return message

    def __str__(self) -> str:
        return json.dumps(
            {
                "text": self.text,
                "role": self.role.value,
                "name": self.name,
                "tags": self.tags,
            },
            indent=4,
        )
