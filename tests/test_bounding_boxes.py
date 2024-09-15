import os
from definitions import PROJECT_DIR
from PIL import Image, ImageDraw
from io import StringIO
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from utils.config_parser import ConfigKey, load_yaml
from actions.action import Action


from pipeline.agent_pipeline import AgentPipeline
from unittest.mock import MagicMock, patch
from bounding_boxes.bounding_box_utils import draw_bounding_boxes


# Load YAML configuration before every test
@pytest.fixture(autouse=True)
def load_config():
    load_yaml("tests/vwa_config.yaml")


@pytest.fixture
def logger():
    return MagicMock()


@pytest.fixture
def vwa_env_name():
    return "vwa_config_files/test_shopping/14.json"


def test_draw_bounding_boxes(vwa_env_name):
    test_files_dir = "tests/bounding_box_files"
    df = pd.read_csv(
        os.path.join(PROJECT_DIR, test_files_dir, "vwa_14_gt_bounding_boxes.csv"),
        delimiter=",",
        quotechar='"',
    )
    bounding_boxes_csv = df.to_csv()
    image = Image.open(os.path.join(PROJECT_DIR, test_files_dir, "vwa_14_1.png"))
    som_image, id2center, _, _ = draw_bounding_boxes(
        bounding_boxes_csv,
        image,
        1,
        viewport_size={"width": 1280, "height": 720},
        window_bounds={
            "upper_bound": 0,
            "left_bound": 0,
            "right_bound": 1280,
            "lower_bound": 720,
        },
    )
    # som_image.show()

    image = Image.open(os.path.join(PROJECT_DIR, test_files_dir, "vwa_14_2.png"))
    som_image, id2center, _, _ = draw_bounding_boxes(
        bounding_boxes_csv,
        image,
        1,
        viewport_size={"width": 1280, "height": 720},
        window_bounds={
            "upper_bound": 720,
            "left_bound": 0,
            "right_bound": 1280,
            "lower_bound": 1440,
        },
        img_padding=20,
    )
    # som_image.show()

    # for each id2center, draw a point on the image
    draw = ImageDraw.Draw(som_image)
    for id, center in id2center.items():
        pixel_ratio = 1
        xy = [
            (center[0] * pixel_ratio) - 10,
            (center[1] * pixel_ratio) - 10,
            (center[0] * pixel_ratio) + 10,
            (center[1] * pixel_ratio) + 10,
        ]
        draw.ellipse(xy, fill="red", width=1)

    # som_image.show()
