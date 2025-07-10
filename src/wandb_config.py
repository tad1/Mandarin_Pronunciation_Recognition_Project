# The purpose of this script, is to:
# - provide a single point for configuring wandb
# - handle login to wandb and set wandb paths

# Note, this file has a hidden coupling with `wandb` package
#   1.  It SHOULD be imported before calling wandb functions

# Example import:
# import wandb, wandb_config

import os
from path import RESULT_DIRECTORY, SECRET_ENV
import dotenv
import wandb

WANDB_PATH = os.path.join(RESULT_DIRECTORY, "wandb")
WANDB_ARTIFACT_DIR = os.path.join(WANDB_PATH, "artifacts")
WANDB_DATA_DIR = os.path.join(WANDB_PATH, "data")
WANDB_ENTITY = "fischbach-kamil-pg"
WANDB_PROJECT = "mandarin-pronunciation"

# wandb requires directories to exist
if not os.path.exists(WANDB_PATH):
    os.makedirs(WANDB_PATH)
if not os.path.exists(WANDB_ARTIFACT_DIR):
    os.makedirs(WANDB_ARTIFACT_DIR)
if not os.path.exists(WANDB_DATA_DIR):
    os.makedirs(WANDB_DATA_DIR)

os.environ["WANDB_DIR"] = RESULT_DIRECTORY
os.environ["WANDB_ARTIFACT_DIR"] = WANDB_ARTIFACT_DIR
os.environ["WANDB_DATA_DIR"] = WANDB_DATA_DIR
os.environ["WANDB_ENTITY"] = WANDB_ENTITY
os.environ["WANDB_PROJECT"] = WANDB_PROJECT

# for WANDB_API_KEY
dotenv.load_dotenv(SECRET_ENV)
if os.environ.get("WANDB_API_KEY") is None:
    raise ValueError(
        f"WANDB_API_KEY is not set in {SECRET_ENV}. "
        "you can get it from https://wandb.ai/settings -> API keys"
    )

wandb.login()