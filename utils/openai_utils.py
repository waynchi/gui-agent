from definitions import PROJECT_DIR
import os
import json
from openai import OpenAI
import base64
from io import BytesIO


def create_client():
    with open(os.path.join(PROJECT_DIR, "api_config.json")) as config_file:
        config = json.load(config_file)
    client = OpenAI(
        api_key=config["openai_api_personal_key"],
    )

    return client


def encode_image_file(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string


def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue())
    return encoded_string
