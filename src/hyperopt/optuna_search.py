# src/models/hyperopt/optuna_search.py

from typing import Dict, Any, Callable
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator
import numpy as np


def build_optuna_objective(
    X, y,
    model: BaseEstimator,
    search_space: Dict[str, Any],
    cv_folds: int
) -> Callable:

    def objective(trial):

        params = {}

        for param_name, spec in search_space.items():

            ptype = spec["type"]

            if ptype == "int":
                params[param_name] = trial.suggest_int(
                    param_name, spec["low"], spec["high"]
                )

            elif ptype == "float":
                params[param_name] = trial.suggest_float(
                    param_name, spec["low"], spec["high"]
                )

            elif ptype == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, spec["values"]
                )

        m = model.__class__(**{**model.get_params(), **params})

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            m.fit(X[train_idx], y[train_idx])
            preds = m.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], preds, average="macro"))

        return np.mean(scores)

    return objective


def run_optuna(
    X, y,
    model: BaseEstimator,
    search_space: Dict[str, Any],
    cv_folds: int,
    n_trials: int = 30
):
    objective = build_optuna_objective(X, y, model, search_space, cv_folds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value
