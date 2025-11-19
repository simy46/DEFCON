# src/models/hyperopt/grid_random_search.py

from typing import Dict, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator

from config.consts import DEFAULT_SCORING_METRIC

def build_grid_search(
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    cv_folds: int,
    scoring: str = DEFAULT_SCORING_METRIC,
    n_jobs: int = -1,
    verbose: int = 2
) -> GridSearchCV:
    return GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )


def build_random_search(
    model: BaseEstimator,
    param_dist: Dict[str, Any],
    n_iter: int,
    cv_folds: int,
    scoring: str = DEFAULT_SCORING_METRIC,
    n_jobs: int = -1,
    verbose: int = 2,
    random_state: int = 42
) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        refit=True
    )
