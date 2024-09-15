import time
import ast
import hashlib
import traceback
from urllib.parse import urlparse
import re
import requests
import shutil
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

# COMCRAWL_CSV = "/data/waynechi/comcrawl_data/comcrawl_2.csv"
# COMCRAWL_SITES_CSV = "/data/waynechi/comcrawl_data/comcrawl_2_sites.csv"
COMCRAWL_CSV = "/mnt/sda/waynechi/comcrawl_data/comcrawl_2.csv"
COMCRAWL_POPULAR_CSV = "/mnt/sda/waynechi/comcrawl_data/comcrawl_2_popular.csv"
COMCRAWL_RANDOM_CSV = "/mnt/sda/waynechi/comcrawl_data/comcrawl_2_random.csv"


class WebCrawler:
    def __init__(self, env: WebArenaEnvironment, data_saver: DataSaver):
        self.env = env
        self.data_saver = data_saver
        self.page_id = 0
        self.visited_urls = set()
        self.max_count = 1000

    def get_page_id(self):
        self.page_id += 1
        return self.page_id

    def get_base_urls(self):
        config_files_dir = "vwa_config_files"
        config_files = os.listdir(config_files_dir)

        config_files_dir = "vwa_config_files"
        # website_types = ["classifieds", "reddit", "shopping"]
        website_types = ["reddit"]
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

    def save_site(self, url, base_url, base_config_file):
        """
        Crawl a single URL
        """
        self.data_saver.start_run(
            "page_{}".format(self.get_page_id()), 0, Task("crawl", [])
        )
        # Navigate to the page
        self.env.env.page.goto(url)

        # Give the page a few seconds to load
        time.sleep(2)

        step_id = 0
        self.save_page(url, base_url, base_config_file, step_id)

        # Handle scroll downs
        previous_window_bounds = self.get_window_bounds()
        self.scroll_down()

        num_scrolls = 0
        max_scrolls = 10
        while (
            previous_window_bounds != self.get_window_bounds()
            and num_scrolls < max_scrolls
        ):
            step_id += 1
            self.save_page(url, base_url, base_config_file, step_id)
            previous_window_bounds = self.get_window_bounds()
            self.scroll_down()
            num_scrolls += 1


def hash_filename(basename):
    # Use SHA-1 to hash the filename
    hash_object = hashlib.sha1(basename.encode())
    hashed_filename = (
        hash_object.hexdigest() + ".html"
    )  # Append the appropriate file extension if needed
    return hashed_filename


class WarcHandler:
    def __init__(self) -> None:
        pass

    def create_comcrawl_dataset(self, sites):
        cdx = cdx_toolkit.CDXFetcher()
        max_records = 5000
        dfs = []
        for site in sites:
            query = {
                "limit": max_records,
                "filter": [
                    "status:200",
                    "mime:text/html",
                ],
            }
            print("Querying site: {}".format(site))
            cdx_iter = cdx.iter(site, **query)
            print("Finished querying site: {}".format(site))
            records = list(cdx_iter)
            print("Finished converting to list")
            # Ensure you don't try to sample more records than exist
            num_records_to_sample = min(100, len(records))

            # Get 100 random records, if there are at least 100 records
            random.seed(77)
            records = random.sample(records, num_records_to_sample)

            data = []
            for record in records:
                data.append(
                    {
                        "warc_url": record["filename"],
                        "warc_filename": record["filename"].split("/")[-1],
                        "warc_folder": record["filename"].split("/")[-1].split(".")[0],
                        "offset": int(record["offset"]),
                        "length": int(record["length"]),
                        "urlkey": record["urlkey"],
                        "timestamp": record["timestamp"],
                        "url": record["url"],
                        "mime": record["mime"],
                        "status": record["status"],
                        "digest": record["digest"],
                        "length": record["length"],
                        "content": record.content,
                    }
                )

            dfs.append(pd.DataFrame(data))

        df = pd.concat(dfs)
        return df

    def create_random_comcrawl_dataset(self):
        cdx = cdx_toolkit.CDXFetcher()

        query = {
            "limit": 50000,
            "filter": [
                "status:200",
                "mime:text/html",
            ],
        }

        print("Querying random webpages from Common Crawl...")
        site = ".com/*"
        print("Querying site: {}".format(site))
        cdx_iter = cdx.iter(site, **query)
        print("Finished querying site: {}".format(site))
        records = list(cdx_iter)
        print("Finished converting to list")

        # Ensure you don't try to sample more records than exist
        num_records_to_sample = min(2500, len(records))
        # Get 100 random records, if there are at least 100 records
        random.seed(77)
        records = random.sample(records, num_records_to_sample)

        data = []
        for record in records:
            data.append(
                {
                    "warc_url": record["filename"],
                    "warc_filename": record["filename"].split("/")[-1],
                    "warc_folder": record["filename"].split("/")[-1].split(".")[0],
                    "offset": int(record["offset"]),
                    "length": int(record["length"]),
                    "urlkey": record["urlkey"],
                    "timestamp": record["timestamp"],
                    "url": record["url"],
                    "mime": record["mime"],
                    "status": record["status"],
                    "digest": record["digest"],
                    "content": record.content,
                }
            )

        df = pd.DataFrame(data)
        return df

    def get_website_directory(self, warc_folder, target_url):
        return os.path.join(warc_folder, "websites", target_url.replace("/", "_"))

    def setup_warc(
        self, warc_url, warc_filename, warc_folder, target_url, offset, length, content
    ):
        # warc_path = os.path.join(warc_folder, warc_filename)
        # # check if file exists
        # if not os.path.isfile(warc_path):
        #     self.download_warc_file(warc_url, warc_filename, warc_path)

        try:
            website_directory = self.get_website_directory(warc_folder, target_url)
            os.makedirs(website_directory, exist_ok=True)
            filename = "index.html"
            filepath = os.path.join(website_directory, filename)
            content = ast.literal_eval(content)
            with open(filepath, "wb") as f:
                f.write(content)
        except OSError as e:
            print(traceback.print_exc())
            print("Attempting to shorten the website directory")
            website_directory = self.get_website_directory(
                warc_folder, hash_filename(target_url)
            )
            os.makedirs(website_directory, exist_ok=True)
            filename = "index.html"
            filepath = os.path.join(website_directory, filename)
            content = ast.literal_eval(content)
            with open(filepath, "wb") as f:
                f.write(content)

        # self.extract_contents_from_warc(
        #     warc_path, website_directory, target_url, offset, length
        # )

    def download_warc_file(self, warc_url, warc_filename, warc_path):
        # download_url = f"https://commoncrawl.s3.amazonaws.com/{warc_filename}"
        download_url = f"https://data.commoncrawl.org/{warc_url}"
        response = requests.get(download_url, stream=True)

        with open(warc_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {warc_filename} to {warc_path}")

    def extract_contents_from_warc(
        self, warc_path, website_directory, target_url, offset, length
    ):
        with open(warc_path, "rb") as stream:
            stream.seek(offset)
            record_stream = BytesIO(stream.read(length))
            found_html = False
            for record in ArchiveIterator(record_stream):
                if record.rec_type == "response":
                    try:
                        content_type = record.http_headers.get_header("Content-Type")
                        target_uri = record.rec_headers.get_header("WARC-Target-URI")
                        content = record.content_stream().read()
                        if "html" in content_type and not found_html:
                            if len(content) > 0:
                                found_html = True
                                filename = "index.html"
                        elif "css" in content_type:
                            filename = target_uri.split("/")[-1] or "styles.css"
                        elif "javascript" in content_type:
                            filename = target_uri.split("/")[-1] or "script.js"
                        filepath = os.path.join(website_directory, filename)
                        with open(filepath, "wb") as f:
                            f.write(content)

                        # if found_html:
                        #     breakpoint()
                        #     break
                    except:
                        continue


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.directory = directory

    def translate_path(self, path):
        # Override the method to set the directory correctly.
        path = super().translate_path(path)
        relpath = os.path.relpath(path, os.getcwd())
        fullpath = os.path.join(self.directory, relpath)
        return fullpath


def run_server(directory, port):
    handler = lambda *args, **kwargs: CustomHTTPRequestHandler(
        *args, directory=directory, **kwargs
    )
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at port {port}")
        httpd.serve_forever()


# def run_server(directory, port, stop_event):
#     handler = lambda *args, **kwargs: CustomHTTPRequestHandler(
#         *args, directory=directory, **kwargs
#     )
#     with socketserver.TCPServer(("", port), handler) as httpd:
#         print(f"Serving at port {port}")
#
#         # This loop checks for a stop event which when set, stops the server
#         while not stop_event.is_set():
#             httpd.handle_request()
#
#         print("Server is stopping.")
#
#
# def restart_server(server_thread, stop_event, directory, port):
#     # Set the event to stop the server
#     stop_event.set()
#     # Wait for the server thread to finish
#     server_thread.join()
#     print("Server has been stopped.")
#
#     # Clear the event and restart the server
#     stop_event.clear()
#     server_thread = threading.Thread(
#         target=run_server, args=(directory, port, stop_event), daemon=True
#     )
#     server_thread.start()
#     print(f"Server restarted on port {port}")
#     return server_thread


def match_site(url, patterns):
    parsed_url = urlparse(url)
    full_url = (
        parsed_url.netloc + parsed_url.path
    )  # Include the network location and path
    # Normalize to handle 'www.'
    if full_url.startswith("www."):
        full_url = full_url[4:]

    for pattern in patterns:
        pattern_regex = pattern.replace(".", "\\.").replace("*", ".*")
        if re.match(pattern_regex, full_url):
            return pattern
    return None


if __name__ == "__main__":
    load_yaml("experiments/default_config.yaml")
    # set_config(ConfigKey.EXPERIMENT_NAME, "vwa_crawl")
    set_config(ConfigKey.EXPERIMENT_NAME, "comcrawl_2")

    data_saver = DataSaver()
    env = WebArenaEnvironment(
        "vwa_config_files/test_classifieds/0.json",
        MagicMock(),
        MagicMock(),
        render_mode=None,
    )
    crawler = WebCrawler(env, data_saver)

    # Crawl common crawl
    warc_handler = WarcHandler()
    sites = [
        # Website details from: https://www.similarweb.com/top-websites/
        # Arts and Entertainment
        "youtube.com/*",
        # "bilibili.com/*",
        "imdb.com/*",
        # "pixiv.net/*",
        "archiveofourown.org/*",
        # "aniwave.to/*",
        # Business and Cosumer Services
        "zillow.com/*",
        "usps.com/*",
        "canadapost-postescanada.ca/*",
        "ups.com/*",
        # "medium.com/*",
        # "fedex.com/*",
        "realtor.com/*",
        "shopify.com/*",
        # Computter Electronics and Technology
        "reddit.com/*",
        "docomo.ne.jp/*",
        "twitter.com/*",
        # Ecommerce and Shopping
        "amazon.com/*",
        "ebay.com/*",
        "rakuten.co.jp/*",
        "aliexpress.com/*",
        "temu.com/*",
        "etsy.com/*"
        # Food and Drink
        "trilltrill.jp/*",
        "cookpad.com/*",
        "tabelog.com/*",
        # "allrecipes.com/*",
        "hotpepper.jp/*",
        # Health
        "nih.gov/*",
        "healthline.com/*",
        "mayoclinic.org/*",
        "webmd.com/*",
        # Hobbies and Leisure
        "shutterstock.com/*",
        "flickr.com/*",
        # "ancestry.com/*",
        "istockphoto.com/*",
        "pixabay.com/*",
        # Home and Garden
        "ikea.com/*",
        "homedepot.com/*",
        "lowes.com/*",
        # "harborfreight.com/*",
        "goodhousekeeping.com/*"
        # Jobs and Career
        "indeed.com/*",
        # "myworkdayjobs.com/*",
        # "hh.ru/*",
        # "computrabajo.com/*"
        # Sports
        "espn.com/*",
        "cricbuzz.com/*",
        "marca.com/*",
        # Travel and Tourism
        "booking.com/*",
        # "agoda.com/*",
        "tripadvisor.com/*",
        "airbnb.com/*",
        "expedia.com/*",
        # Past
        # "google.com/*",
        # "youtube.com/*",
        # "facebook.com/*",
        # "instagram.com/*",
        # "twitter.com/*",
        # "pinterest.com/*",
        # "linkedin.com/*",
        # "reddit.com/r/*",
        # "amazon.com/*",
        # "ebay.com/*",
        # "wikipedia.com/*",
        # "fandom.com/*",
        # "quora.com/*",
    ]
    print("Number of sites: {}".format(len(sites)))
    print("Getting common crawl dataset")
    if not os.path.exists(COMCRAWL_CSV):
        # if True:
        if not os.path.exists(COMCRAWL_POPULAR_CSV):
            popular_df = warc_handler.create_comcrawl_dataset(sites)
            popular_df.to_csv(COMCRAWL_POPULAR_CSV, index=False)
        else:
            popular_df = pd.read_csv(COMCRAWL_POPULAR_CSV)
        if not os.path.exists(COMCRAWL_RANDOM_CSV):
            random_df = warc_handler.create_random_comcrawl_dataset()
            random_df.to_csv(COMCRAWL_RANDOM_CSV, index=False)
        else:
            random_df = pd.read_csv(COMCRAWL_RANDOM_CSV)
        df = pd.concat([popular_df, random_df])
        # df = popular_df
        df.to_csv(COMCRAWL_CSV, index=False)
    else:
        df = pd.read_csv(COMCRAWL_CSV)

    print("Serving common crawl website")

    # Limit dataframe to include only the first 10 records per site
    # df["site_pattern"] = df["url"].apply(lambda x: match_site(x, sites))
    # # Filter out any rows where 'site_pattern' is None
    # df_filtered = df[df["site_pattern"].notna()]
    # # Group by 'site_pattern' and grab the first 10 entries for each
    # df_top10 = df_filtered.groupby("site_pattern").head(10)
    # df = df_top10

    for (
        warc_url,
        warc_filename,
        warc_folder,
        target_url,
        offset,
        length,
        content,
    ) in zip(
        df["warc_url"],
        df["warc_filename"],
        df["warc_folder"],
        df["url"],
        df["offset"],
        df["length"],
        df["content"],
    ):
        print(
            "warc_url: {} warc_filename: {} warc_folder: {} target_url: {} offset: {} length: {}".format(
                warc_url,
                warc_filename,
                warc_folder,
                target_url,
                int(offset),
                int(length),
            )
        )
        try:
            warc_handler.setup_warc(
                warc_url,
                warc_filename,
                "/mnt/sda/waynechi/comcrawl_data",
                target_url,
                int(offset),
                int(length),
                content,
            )
        except Exception as e:
            import traceback

            print(traceback.print_exc())
            breakpoint()
            print("Next Line")

    # stop_event = threading.Event()
    server_thread = threading.Thread(
        target=run_server,
        args=("./", 8000),
        daemon=True,
    )
    server_thread.start()

    # idx = 1374
    # idx = 1861
    idx = 3500
    to_idx = 3800
    breakpoint()
    # Drop the first idx amount of websites
    df = df.iloc[idx:to_idx]
    print(len(df))

    for warc_url, warc_filename, warc_folder, target_url, offset, length in zip(
        df["warc_url"],
        df["warc_filename"],
        df["warc_folder"],
        df["url"],
        df["offset"],
        df["length"],
    ):
        try:
            website_directory = os.path.join(
                "/mnt/sda/waynechi/comcrawl_data/websites", target_url.replace("/", "_")
            )
            if not os.path.exists(website_directory):
                # Try hashing
                website_directory = os.path.join(
                    "/mnt/sda/waynechi/comcrawl_data/websites",
                    hash_filename(target_url).replace("/", "_"),
                )
            print("Website directory: {}".format(website_directory))
            # Make a local directory
            local_directory = "website_{}".format(idx)
            shutil.copytree(website_directory, local_directory, dirs_exist_ok=True)
            url = "http://localhost:8000/{}".format(local_directory)
            print("url: {}".format(url))
            crawler.save_site(url, target_url, None)
            # Delete the local directory
            idx += 1
            shutil.rmtree(local_directory)
        except Exception as e:
            import traceback

            print(traceback.print_exc())
            continue
        # server_thread = restart_server(server_thread, stop_event, "./", 8000)

    # crawler = WebCrawler(env, data_saver)

    # max_count = 50
    # for site in sites:
    #     site_count = 0
    #     for url in df["url"]:
    #         if site_count > max_count:
    #             break
    #         if site.replace("*", "") in url:
    #             crawler.save_site(url, url, None)
    #             site_count += 1
