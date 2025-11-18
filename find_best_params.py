# HOW TO RUN :
# python find_best_params.py --config config/model_lr.yaml (state the path of models config file)

# THIS FILES PURPOSE IS TO FIND THE BEST HYPERPARAMS OF A MODEL
# AFTER FINDING THE BEST PARAMS, YOU SHOULD RUN MAIN.PY WITH THE NEW .YAML FILE 


import argparse
import yaml

from src.utils.logging_utils import get_logger
from src.features.preprocessing import apply_preprocessing
from src.utils.timer import Timer
from src.data.data_loader import load_test, load_train
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

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
    X_test, metadata_test = load_test()


# -----------------------------------
# Preprocessing
# -----------------------------------
with Timer("Preprocessing data..."):
    X_train_prep, X_test_prep = apply_preprocessing(
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
assert model_cfg["name"] == "random_forest"

rf = RandomForestClassifier(
    n_jobs=model_cfg.get("n_jobs", -1),
    random_state=model_cfg.get("random_state", 42)
)


# -----------------------------------
# HYPERPARAMETER SEARCH SPACE
# -----------------------------------
param_grid = {
    "n_estimators": [100, 300, 500, 800, 1200],
    "max_depth": [None, 6, 10, 20, 40],
    "min_samples_split": [2, 5, 10, 20, 40],
    "min_samples_leaf": [1, 2, 4, 10, 20],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 1.0],
    "criterion": ["gini", "entropy"],
    "bootstrap": [True, False]
}


# -----------------------------------
# RandomizedSearch or GridSearch
# -----------------------------------
SEARCH_MODE = "random"  # or "grid"

logger.info(f"Running hyperparameter search: {SEARCH_MODE}")

if SEARCH_MODE == "grid":
    search = GridSearchCV(
        rf,
        param_grid=param_grid,
        cv=model_cfg["cv_folds"],
        verbose=3,
        n_jobs=-1
    )
else:
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_grid,
        n_iter=40,                 # Number of random combinations
        cv=model_cfg["cv_folds"],
        verbose=3,
        n_jobs=-1,
        random_state=42,
        scoring="f1"
    )


# -----------------------------------
# Run search
# -----------------------------------
with Timer("Hyperparameter search..."):
    search.fit(X_train_prep, y_train)


logger.info(f"Best score: {search.best_score_}")
logger.info(f"Best params: {search.best_params_}")


# -----------------------------------
# Save best params back to the config file
# -----------------------------------
best_params = search.best_params_
cfg["model"].update(best_params)

output_file = f"{args.config.replace('.yaml', '')}_best.yaml"
with open(output_file, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)
