# This code is responsible to handle coupling between code and paths
import os

PROJECT_ROOT_DIRECTORY = CWD = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_DIRECTORY = os.path.join(PROJECT_ROOT_DIRECTORY, "src")
RESULT_DIRECTORY = os.path.join(PROJECT_ROOT_DIRECTORY, "res")
DATA_DIRECTORY = os.path.join(PROJECT_ROOT_DIRECTORY, "data")
DOCUMENTATION_DIRECTORY = os.path.join(PROJECT_ROOT_DIRECTORY, "doc")

# source dataset paths
PG_EXPERIMENT_PATH = os.path.join(DATA_DIRECTORY, "source/pg_dataset/")
## requires dataset structure:
    # PG_EXPERIMENT_PATH
    # ├── assesment.csv
    # ├── experiment.csv
    # ├── recordings
    # │   ├── stageI
    # │   │   └── {id}
    # │   │       └── {rec_id}.ogg
    # │   └── stageII
    # │       └── {id}
    # │           └── {rec_id}.ogg
    # └── tones_with_label.xls
    

COMMON_VOICE_PATH= os.path.join(PROJECT_ROOT_DIRECTORY, "../../../Data/cv-corpus-20.0-2024-12-06/zh-CN/")