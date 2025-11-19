from __future__ import annotations
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier

UNWANTED_MODEL_KEYS = {"name", "cv_folds"}
UNWANTED_SEARCH_KEYS = {"n_iter", "n_trials"}


def build_model(
    model_cfg: Dict[str, Any],
    hyper_cfg: Dict[str, Any]
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Build a RandomForestClassifier model using model_cfg, excluding
    hyperparameters that will be optimized in hyper_cfg.

    Returns:
        model (RandomForestClassifier)
        search_space (Dict[str, Any]) - only params to optimize
    """

    mode: str = hyper_cfg["type"]
    section: Dict[str, Any] = hyper_cfg[mode]

    search_space: Dict[str, Any] = {
        param: values
        for param, values in section.items()
        if param not in UNWANTED_SEARCH_KEYS and values is not None
    }

    base_params: Dict[str, Any] = {
        key: value
        for key, value in model_cfg.items()
        if key not in search_space and key not in UNWANTED_MODEL_KEYS
    }

    model = RandomForestClassifier(**base_params)

    return model, search_space
