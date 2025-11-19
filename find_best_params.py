# HOW TO RUN :
# python find_best_params.py --config config/model_lr.yaml (state the path of models config file)

# THIS FILES PURPOSE IS TO FIND THE BEST HYPERPARAMS OF A MODEL
# AFTER FINDING THE BEST PARAMS, YOU SHOULD RUN MAIN.PY WITH THE NEW .YAML FILE 


import argparse
import yaml

from config.consts import GRID_SEARCH_MODE, OPTUNA_MODE, RANDOM_MODE
from src.utils.logging_utils import get_logger
from src.features.preprocessing import apply_preprocessing
from src.utils.timer import Timer
from src.data.data_loader import load_test, load_train
from hyperopt.build_hyper_model import build_model
from hyperopt.build_search import build_search
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# -----------------------------------
# Parse command line arguments
# -----------------------------------
parser = argparse.ArgumentParser(
    description="Run the full ML pipeline. Example:\n"
                "  python main.py --config config/model_rf.yaml\n"
                "Available model configs:\n"
                "  - config/model_rf.yaml\n"
                "  - config/model_svm.yaml\n"
                "  - config/model_xgb.yaml\n",
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the YAML config file (ex: config/model_rf.yaml)"
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
    X_test, metadata_test = load_test()


# -----------------------------------
# Preprocessing
# -----------------------------------
with Timer("Preprocessing data..."): 
    X_train_prep, X_test_prep = apply_preprocessing( # here we assume that the data preprocessing step is optimized
        X_train=X_train, 
        X_test=X_test, 
        y_train=y_train,
        metadata_train=metadata_train, 
        metadata_test=metadata_test, 
        cfg=cfg
    )


# -----------------------------------
# Build Model
# -----------------------------------
model_cfg = cfg["model"]
hyperop_cfg = cfg["hyperoptimization"]
assert model_cfg["name"] == "random_forest" # we could work on xgboost too


# -----------------------------------
# MODEL & HYPERPARAMETER SEARCH SPACE
# -----------------------------------
rf, search_space = build_model(model_cfg, hyperop_cfg)

# -----------------------------------
# Determine optimization mode
# -----------------------------------
mode = cfg["hyperoptimization"]["type"]

# -----------------------------------
# RandomizedSearch, GridSearch or Optuna
# -----------------------------------
search = build_search(
    model=rf,
    search_space=search_space,
    cfg=cfg,
    X_train=X_train_prep,
    y_train=y_train
)

# -----------------------------------
# Run search
# -----------------------------------
if mode in (RANDOM_MODE, GRID_SEARCH_MODE):
    with Timer("Hyperparameter search..."):
        search.fit(X_train_prep, y_train)

    best_score = search.best_score_
    best_params = search.best_params_

elif mode == OPTUNA_MODE:
    best_params, best_score = search


logger.info(f"Best score: {best_score}")
logger.info(f"Best params: {best_params}")


# -----------------------------------
# Save best params back to the config file
# -----------------------------------
cfg["model"].update(best_params)

output_file = f"{args.config.replace('.yaml', '')}_best.yaml"
with open(output_file, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)
