from typing import Dict, Any, Callable, Tuple
import optuna
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def build_optuna_objective(
    X, y,
    model: BaseEstimator,
    search_space: Dict[str, Any],
    cv_folds: int
) -> Callable:

    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)

    def objective(trial):

        params = {}

        for name, spec in search_space.items():

            t = spec["type"]

            if t == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])

            elif t == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])

            elif t == "categorical":
                params[name] = trial.suggest_categorical(name, spec["values"])

        m = clone(model)
        m.set_params(**params)

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42
        )

        scores = []

        for train_idx, val_idx in cv.split(X, y):
            try:
                m.fit(X[train_idx], y[train_idx])
                preds = m.predict(X[val_idx])
                scores.append(
                    f1_score(y[val_idx], preds, average="macro")
                )

            except Exception:
                return -1e9

        return float(np.mean(scores))

    return objective



def run_optuna(
    X, y,
    model: BaseEstimator,
    search_space: Dict[str, Any],
    cv_folds: int,
    n_trials: int = 30
) -> Tuple[Dict[str, Any], float]:

    objective = build_optuna_objective(
        X, y,
        model=model,
        search_space=search_space,
        cv_folds=cv_folds
    )

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    return study.best_params, study.best_value
