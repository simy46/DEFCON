# HOW TO RUN :
# python main.py --config config/model_lr.yaml

# WHAT IT DOES :
# 1. Loads the YAML config (cfg is a Python dict)
# 2. Loads data via the data-loader (auto-decompress if needed)
# 3. Applies preprocessing (PCA, scaling, etc.)
# 4. Builds the model from the config, trains it, and makes predictions
# 5. Generates submission.csv for Kaggle


import argparse
import yaml
import pandas as pd

from src.utils.logging_utils import get_logger
from src.data.data_loader import load_train, load_test
from src.features.preprocessing import apply_preprocessing
from src.models.train_model import train_model
from src.models.predict import predict_model
from src.utils.timer import Timer


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

logger = get_logger()
logger.info(f"Loaded config: {args.config}")

# -----------------------------------
# Load training and test data
# -----------------------------------
with Timer("Loading train data..."):
    X_train, metadata_train, y_train = load_train()

with Timer("Loading test data..."):
    X_test, metadata_test = load_test()

# -----------------------------------
# Preprocessing
# -----------------------------------
with Timer("Preprocessing data..."):
    X_train, X_test = apply_preprocessing(X_train, X_test, cfg)

# -----------------------------------
# Train model
# -----------------------------------
with Timer("Training model..."):
    model = train_model(X_train, y_train, cfg)

# -----------------------------------
# Predict on test
# -----------------------------------
with Timer("Predicting on test set..."):
    preds = predict_model(model, X_test, metadata_test)

# -----------------------------------
# Save submission file
# -----------------------------------
logger.info("Saving submission.csv...")
submission = pd.DataFrame({
    "label": preds.astype(int)   # Kaggle requires 0/1 integers
})
submission.to_csv("submission.csv", index=False)
logger.info("Pipeline completed successfully.")