import yaml
from enum import Enum

# Global variable to hold the YAML data
yaml_data = None
default_yaml_data = None


class ConfigKey(str, Enum):
    EXPERIMENT_NAME = "experiment_name"
    ZENO_PROJECT_NAME = "zeno_project_name"
    E2E_MAX_RETRIES = "e2e_max_retries"
    RATE_LIMIT_MAX_RETRIES = "rate_limit_max_retries"
    AGENT_PREDICT_MAX_RETRIES = "agent_predict_max_retries"
    MODEL_NAME = "model_name"
    PROVIDER = "provider"
    ACTION_TYPES = "action_types"
    TOKEN_LIMIT = "token_limit"
    ENVIRONMENT_TYPE = "environment_type"
    PROMPT_GENERATOR_TYPE = "prompt_generator_type"
    RESPONSE_PARSER_TYPE = "response_parser_type"
    STATE_TYPE = "state_type"
    AGENT_TYPE = "agent_type"
    MAX_STEPS = "max_steps"
    MERGE_IOU_THRESHOLD = "merge_iou_threshold"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    REPEAT_TOLERANCE = "repeat_tolerance"
    WEBSITE_TYPES = "website_types"
    INSTRUCTION_PATH = "instruction_path"
    FILTER_INSTRUCTION_PATH = "filter_instruction_path"
    BBOX_MODEL_PATH = "bbox_model_path"
    USE_BBOXES = "use_bboxes"
    USE_TAGS = "use_tags"
    USE_IDS = "use_ids"
    USE_INTERACT_ELEMENT_TEXT = "use_interact_element_text"
    USE_STATIC_TEXT = "use_static_text"
    ALLOW_INTERACTION_WITH_TEXT = "allow_interaction_with_text"
    ENV_START_IDX = "env_start_idx"
    NUM_ENVIRONMENTS = "num_environments"
    USE_ALL_STATIC_TEXT = "use_all_static_text"
    USE_ORDERING = "use_ordering"
    TSNE_PERPLEXITY = "tsne_perplexity"
    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    PROMPT_MODE = "prompt_mode"
    DATASET_PARTITION = "dataset_partition"
    RANDOM_SEED = "random_seed"
    NO_IMG = "no_img"
    OCR_ONLY = "ocr_only"


def load_yaml(file_path):
    """Load the YAML file and store its contents in a global variable."""
    global yaml_data
    global default_yaml_data
    with open(file_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    with open("experiments/default_config.yaml", "r") as file:
        default_yaml_data = yaml.safe_load(file)


def get_yaml_data():
    """Return the loaded YAML data."""
    return yaml_data, default_yaml_data


def get_config(key: ConfigKey, default=None):
    result = yaml_data.get(key, default)
    if result is None:
        # TODO log if picking from default?
        result = default_yaml_data.get(key, None)

    if result is None:
        raise Exception(
            "Key {} not found in experiment yaml or default_config.yaml".format(key)
        )

    return result


def set_config(key: ConfigKey, value):
    yaml_data[key] = value
