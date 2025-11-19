from typing import Dict, Any
from sklearn.base import BaseEstimator

from .grid_random_search import build_grid_search, build_random_search
from .optuna_search import run_optuna
from config.consts import GRID_SEARCH_MODE, RANDOM_MODE, OPTUNA_MODE, DEFAULT_SCORING_METRIC

def build_search(
    model: BaseEstimator,
    search_space: Dict[str, Any],
    cfg: Dict[str, Any],
    X_train=None,
    y_train=None
):
    hyper = cfg["hyperoptimization"]
    mode = hyper["type"]
    cv_folds = cfg["model"]["cv_folds"]

    if mode == GRID_SEARCH_MODE:
        return build_grid_search(
            model=model,
            param_grid=search_space,
            cv_folds=cv_folds,
            scoring=DEFAULT_SCORING_METRIC,
            n_jobs=-1,
            verbose=3
        )

    elif mode == RANDOM_MODE:
        return build_random_search(
            model=model,
            param_dist=search_space,
            n_iter=hyper["random"]["n_iter"],
            cv_folds=cv_folds,
            scoring=DEFAULT_SCORING_METRIC,
            n_jobs=-1,
            verbose=3
        )

    elif mode == OPTUNA_MODE:
        if X_train is None or y_train is None:
            raise ValueError("Optuna requires X_train and y_train")

        best_params, best_score = run_optuna(
            X=X_train,
            y=y_train,
            model=model,
            search_space=search_space,
            cv_folds=cv_folds,
            n_trials=hyper["optuna"]["n_trials"]
        )
        return best_params, best_score

    else:
        raise ValueError(f"Unknown optimization mode: {mode}")
