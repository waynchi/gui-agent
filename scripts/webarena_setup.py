#!/usr/bin/env python3
# type: ignore
"""
Link this file to the root directory of webarena. Example:
ln -s ~/dev/scripts/webarena_setup.py ~/dev/webarena/webarena_setup.py

Run this file from the root directory of webarena.
"""

import json
import os
import re
import subprocess
import time

SLEEP = 1.5
# set the URLs of each website, we use the demo sites as an example
os.environ["SHOPPING"] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
os.environ["SHOPPING_ADMIN"] = (
    "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
)
os.environ["REDDIT"] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
os.environ["GITLAB"] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
os.environ["MAP"] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
os.environ["WIKIPEDIA"] = (
    "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
)
os.environ["HOMEPAGE"] = (
    "PASS"  # The home page is not currently hosted in the demo site
)
print("Done setting up URLs")

# First, run `python scripts/generate_test_data.py` to generate the config files
p = subprocess.run(["python", "scripts/generate_test_data.py"], capture_output=True)

# It will generate individual config file for each test example in config_files
assert os.path.exists("config_files/0.json")

# Make sure the URLs in the config files are replaced properly
with open("config_files/0.json", "r") as f:
    config = json.load(f)
    assert os.environ["SHOPPING_ADMIN"] in config["start_url"], (
        os.environ["SHOPPING_ADMIN"],
        config["start_url"],
    )

print("Done generating config files with the correct URLs")
subprocess.run(["cp", "-r", "config_files/", "../gui-agent/config_files"])

# run bash prepare.sh to save all account cookies, this only needs to be done once
# subprocess.run(["bash", "prepare.sh"])
# print("Done saving account cookies")

# Copy files over to the gui-agent directory
# subprocess.run(["cp", "-r", ".auth/", "../gui-agent/.auth"])
