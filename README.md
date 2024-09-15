# Overview

This repository exists for the purposes of reproducibility. In addition to all of the source code, there are a few portions we believe would provide the most value.

## GUI Element Deteciton Model

The best GUI element detection model we trained can be found at bounding_boxes/outputs/comcrawl_200k_lr_0.0025_unfrozen/model_final.pth

## Experimental Settings

All experimental settings can be found in the experiments folder

After Setup, experiments can be run with (replace with your experiment):
```
python3 main.py --config_file experiments/final_omniact_gemini_gt_bbox_gt_order_all.yaml
```

More examples can be found in `run_local_scripts.sh`

## Prompts

All prompts can be found at: `prompt_generators/example_prompts/`

# Setup

```
conda create -n llm-agent python=3.10 -y
conda activate llm-agent
```

### Add to PYTHONPATH

```
export PYTHONPATH='~/gui-agent/':$PYTHONPATH
source ~/.bash_profile
```

### Linux
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Mac
conda install pytorch torchvision torchaudio cpuonly -c pytorch


### Requirements
```
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git transformers einops 
# For webarena
pip install --upgrade google-cloud-aiplatform 
```

### Install Decatron
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

### VisualWebArena

Follow the instructions at https://github.com/web-arena-x/visualwebarena. This is necessary for the agent to run due to imports.
```
cd visualwebarena
pip install -r requirements.txt
playwright install
pip install -e .
```

Change the web_arena_prepare.sh file to map to your VWA install directory. Run it. This is only necessary for VWA experiments.

Note that due to differing OpenAI versions, you may have to go into webarena and manually fix the outdated APIs.
The error classes will look something like openai.****.RateLimitingError. Change it to openai.RateLimitngError

# Gcloud
If you're using Gemini, set up your GCP.
```
gcloud auth login
gcloud config set project YOUR-PROJECT
gcloud auth application-default login
```

### Run this
import nltk
nltk.download('punkt')