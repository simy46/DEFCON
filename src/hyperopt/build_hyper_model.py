from __future__ import annotations
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

UNWANTED_MODEL_KEYS = {"name", "cv_folds"}
UNWANTED_SEARCH_KEYS = {"n_iter", "n_trials"}

MODEL_FACTORY = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "randomized_tree": ExtraTreesClassifier
}


def build_model(
    model_cfg: Dict[str, Any],
    hyper_cfg: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:

    mode: str = hyper_cfg["type"]
    section: Dict[str, Any] = hyper_cfg[mode]

    search_space: Dict[str, Any] = {
        param: values
        for param, values in section.items()
        if param not in UNWANTED_SEARCH_KEYS
        and isinstance(values, dict)
    }

    base_params: Dict[str, Any] = {
        key: value
        for key, value in model_cfg.items()
        if key not in search_space and key not in UNWANTED_MODEL_KEYS
    }

    name = model_cfg["name"]

    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model '{name}'")

    ModelClass = MODEL_FACTORY[name]

    model = ModelClass(**base_params)

    return model, search_space