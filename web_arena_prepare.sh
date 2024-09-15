#!/bin/bash

# prepare the evaluation
# re-validate login information
mkdir -p ./.auth
# python ../webarena/browser_env/auto_login.py
python ../visualwebarena/browser_env/auto_login.py
