import requests
import random
import http.server
import socketserver
import threading
from warcio.archiveiterator import ArchiveIterator
from PIL import Image
import json
import os
from io import StringIO, BytesIO
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse
from utils.data_saver import DataSaver
from utils.config_parser import load_yaml, ConfigKey, set_config
from bounding_boxes.bounding_box_utils import draw_bounding_boxes
from browser_env import create_scroll_action
from environments.webarena_environment import WebArenaEnvironment
from unittest.mock import MagicMock
import cdx_toolkit
from environments.task import Task
import pandas as pd

COMCRAWL_CSV = "/data/waynechi/comcrawl_data/comcrawl.csv"


class WebCrawler:
    def __init__(self, env: WebArenaEnvironment, data_saver: DataSaver):
        self.env = env
        self.MAX_COUNT_PER_TYPE = 1000
        self.data_saver = data_saver
        self.page_id = 1
        self.visited_urls = set()
        self.max_count = self.MAX_COUNT_PER_TYPE

    def reset_for_type(self):
        self.max_count += self.MAX_COUNT_PER_TYPE

    def get_page_id(self):
        self.page_id += 1
        return self.page_id

    def get_base_urls(self, website_types):
        config_files_dir = "vwa_config_files"
        config_files = os.listdir(config_files_dir)

        config_files_dir = "vwa_config_files"
        # website_types = ["classifieds", "reddit", "shopping"]
        all_config_files = []
        for website_type in website_types:
            config_files = os.listdir(
                os.path.join(config_files_dir, f"test_{website_type}")
            )

            # Get all the {int}.json files in the config_files directory
            all_config_files.extend(
                [
                    os.path.join(config_files_dir, f"test_{website_type}", file)
                    for file in config_files
                    if file.endswith(".json") and file[0].isdigit()
                ]
            )

        # It is important to go in order of the files since there are some
        # PUT or POST actions that require resetting the environment
        # Going in order prevents the need to reset.
        all_config_files = sorted(
            all_config_files, key=lambda x: int(x.split("/")[-1].split(".")[0])
        )

        base_urls = set(
            [
                json.load(open(config_file, "r"))["start_url"]
                for config_file in all_config_files
            ]
        )
        return zip(base_urls, all_config_files)

    def save_page(self, url, base_url, base_config_file, step_id):
        self.data_saver.start_step(step_id)
        # Taking screenshot into a BytesIO object
        screenshot_bytes = self.env.env.page.screenshot()
        image_stream = BytesIO(screenshot_bytes)
        # Creating a PIL image from the byte stream
        image = Image.open(image_stream)
        self.data_saver.save_base_image(image)

        # Get the bounding boxes and save information
        bounding_boxes_csv = self.env.get_bounding_boxes()

        som_image, id2center, content_str, filtered_df = draw_bounding_boxes(
            bounding_boxes_csv,
            image,
            pixel_ratio=self.env.get_pixel_ratio(),
            viewport_size=self.env.get_viewport_size(),
            window_bounds=self.get_window_bounds(),
        )
        self.data_saver.save_gt_bounding_boxes(filtered_df.to_csv())

        self.data_saver.save_gt_som_image(som_image)
        self.data_saver.save_info(
            {
                "url": url,
                "base_url": base_url,
                "base_config_file": base_config_file,
                "viewport_size": self.env.get_viewport_size(),
                "window_bounds": self.get_window_bounds(),
            }
        )

    def scroll_down(self):
        # scroll_down_action = create_scroll_action("down")
        # self.env.env.step(scroll_down_action)
        self.env.env.page.evaluate(
            """
            window.scrollBy(0, window.innerHeight / 2);
        """
        )

    def get_window_bounds(self):
        # return self.env.get_window_bounds()
        bounds = self.env.env.page.evaluate(
            """
            () => {
                return {
                    upper_bound: window.pageYOffset,
                    lower_bound: window.pageYOffset + window.innerHeight,
                    left_bound: window.pageXOffset,
                    right_bound: window.pageXOffset + window.innerWidth
                };
            }
            """
        )
        return bounds

    def crawl(self, url, base_url, base_config_file, max_depth=3, depth=0):
        # Avoid infinite loops and control recursion depth
        if self.page_id > self.max_count:
            return

        if url is None:
            return

        if url.endswith("/"):
            url = url[:-1]

        if (
            url in self.visited_urls
            or depth > max_depth
            or not url.startswith("http://treble.cs.cmu.edu")
        ):
            return

        self.visited_urls.add(url)

        print(f"Crawling: {url}, Depth: {depth}")

        try:
            self.save_site(url, base_url, base_config_file)
            # Get all links on the page
            links = self.env.env.page.query_selector_all("a")
            hrefs = [link.get_attribute("href") for link in links]

            # Recursively visit each link
            for href in hrefs:
                # Join relative URLs with the base URL
                next_url = (
                    urljoin(base_url, href) if urlparse(href).netloc == "" else href
                )
                self.crawl(next_url, base_url, base_config_file, max_depth, depth + 1)

        except Exception as e:
            print(f"Failed to process {url}: {e}")

    def save_site(self, url, base_url, base_config_file, max_scrolls=10):
        """
        Crawl a single URL
        """
        self.data_saver.start_run(
            "page_{}".format(self.get_page_id()), 0, Task("crawl", [])
        )
        # Navigate to the page
        self.env.env.page.goto(url)

        step_id = 0
        self.save_page(url, base_url, base_config_file, step_id)

        # Handle scroll downs
        previous_window_bounds = self.get_window_bounds()
        self.scroll_down()

        num_scrolls = 0
        while (
            previous_window_bounds != self.get_window_bounds()
            and num_scrolls < max_scrolls
        ):
            step_id += 1
            self.save_page(url, base_url, base_config_file, step_id)
            previous_window_bounds = self.get_window_bounds()
            self.scroll_down()
            num_scrolls += 1


if __name__ == "__main__":
    load_yaml("experiments/default_config.yaml")
    set_config(ConfigKey.EXPERIMENT_NAME, "vwa_crawl")

    data_saver = DataSaver()
    env = WebArenaEnvironment(
        "vwa_config_files/test_classifieds/0.json",
        MagicMock(),
        MagicMock(),
        render_mode=None,
    )
    crawler = WebCrawler(env, data_saver)

    # Crawl VWA
    base_urls_and_config_files = crawler.get_base_urls(["classifieds"])
    for base_url, config_file in base_urls_and_config_files:
        crawler.crawl(base_url, base_url, config_file, max_depth=3)

    crawler.reset_for_type()
    base_urls_and_config_files = crawler.get_base_urls(["reddit"])
    for base_url, config_file in base_urls_and_config_files:
        crawler.crawl(base_url, base_url, config_file, max_depth=3)

    crawler.reset_for_type()
    base_urls_and_config_files = crawler.get_base_urls(["shopping"])
    for base_url, config_file in base_urls_and_config_files:
        crawler.crawl(base_url, base_url, config_file, max_depth=3)
