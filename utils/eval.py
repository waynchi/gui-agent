import os
from definitions import PROJECT_DIR
import json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

from zeno_client import ZenoClient, ZenoMetric


def create_zeno_project(project_name):
    # read api key from config.json
    with open(os.path.join(PROJECT_DIR, "api_config.json")) as config_file:
        config = json.load(config_file)
    api_key = config["zeno_api_key"]
    client = ZenoClient(api_key)

    project = client.create_project(
        name=project_name,
        view={
            "data": {"type": "text"},
            "label": {"type": "text"},
            "output": {"type": "text"},
        },
        metrics=[
            ZenoMetric(name="accuracy", type="mean", columns=["label"]),
        ],
    )

    return project

def upload_zeno_project(project, data, labels):
        # prompt the user to ensure that they want to create a new project
        print("Are you sure you want to create a new project? (y/n)")
        response = input()
        if response == "y":
            df = pd.DataFrame()
            # add task to df
            df["data"] = data
            df["labels"] = labels
            df["id"] = df.index

            project.upload_dataset(
                df, id_column="id", data_column="data", label_column="labels"
            )
