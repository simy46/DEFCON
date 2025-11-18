# HOW TO RUN :
# python find_best_params.py --config config/model_lr.yaml (state the path of models config file)

# THIS FILES PURPOSE IS TO FIND THE BEST HYPERPARAMS OF A MODEL
# AFTER FINDING THE BEST PARAMS, YOU SHOULD RUN MAIN.PY WITH THE NEW .YAML FILE 


import argparse
import yaml

from src.utils.logging_utils import get_logger
from src.utils.timer import Timer
from src.data.data_loader import load_train


# -----------------------------------
# Parse command line arguments
# -----------------------------------
parser = argparse.ArgumentParser(
    description="Run the full ML pipeline. Example:\n"
                "  python main.py --config config/model_lr.yaml\n"
                "Available model configs:\n"
                "  - config/model_lr.yaml\n"
                "  - config/model_svm.yaml\n"
                "  - config/model_xgb.yaml\n",
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the YAML config file (ex: config/model_lr.yaml)"
)

args = parser.parse_args()

# -----------------------------------
# Load YAML config
# -----------------------------------
with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

logger, timestamp = get_logger()
logger.info(f"Loaded config ({args.config})" + yaml.dump(cfg, sort_keys=False))


# -----------------------------------
# Load training and test data
# -----------------------------------
with Timer("Loading training data..."):
    X_train, metadata_train, y_train = load_train()



# -----------------------------------
# Preprocessing
# -----------------------------------
with Timer("Preprocessing data..."):
    X_train, X_test = apply_preprocessing(
        X_train=X_train, 
        X_test=X_test, 
        y_train=y_train,
        metadata_train=metadata_train, 
        metadata_test=metadata_test, 
        cfg=cfg
    )