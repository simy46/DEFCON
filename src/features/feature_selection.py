from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from xgboost import XGBClassifier # type: ignore


def filter_low_variance_features(
    X_train: NDArray,
    X_test: NDArray,
    threshold: float
) -> Tuple[NDArray, NDArray]:
    """
    Remove features with variance below a threshold.

    Parameters
    X_train : NDArray
        Training feature matrix.
    X_test : NDArray
        Test feature matrix.
    threshold : float
        Minimum variance required to keep a feature.

    Returns
    Tuple[NDArray, NDArray]
        Filtered (X_train, X_test) matrices.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel: NDArray = selector.fit_transform(X_train)
    X_test_sel: NDArray = selector.transform(X_test)
    return X_train_sel, X_test_sel


def select_first_features(
    X_train: NDArray,
    X_test: NDArray,
    n_features: int
) -> Tuple[NDArray, NDArray]:
    """
    Select the first n_features columns.

    Parameters
    X_train : NDArray
        Training feature matrix.
    X_test : NDArray
        Test feature matrix.
    n_features : int
        Number of columns to keep.

    Returns
    Tuple[NDArray, NDArray]
        Reduced (X_train, X_test) matrices.
    """
    X_train_sel: NDArray = X_train[:, :n_features]
    X_test_sel: NDArray = X_test[:, :n_features]
    return X_train_sel, X_test_sel


def select_k_best_features(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    k: int
) -> Tuple[NDArray, NDArray]:
    """
    Select the top-k most informative features (supervised).

    Parameters
    X_train : NDArray
        Training feature matrix.
    y_train : NDArray
        Training labels.
    X_test : NDArray
        Test feature matrix.
    k : int
        Number of best features to keep.

    Returns
    Tuple[NDArray, NDArray]
        (X_train_selected, X_test_selected)
    """
    assert (X_train >= 0).all()
    selector = SelectKBest(f_classif, k=k)

    X_train_sel: NDArray = selector.fit_transform(X_train, y_train)
    X_test_sel: NDArray = selector.transform(X_test)

    return X_train_sel, X_test_sel


def select_topk_xgboost(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    k: int,
    random_state: int = 42
) -> Tuple[NDArray, NDArray]:
    """
    Select top-k most important features using XGBoost feature importance.

    Parameters
    X_train : NDArray
        Training feature matrix.
    y_train : NDArray
        Training labels (0/1).
    X_test : NDArray
        Test feature matrix.
    k : int
        Number of best features to select.
    random_state : int
        Seed for XGBoost reproducibility.

    Returns
    Tuple[NDArray, NDArray]
        (X_train_selected, X_test_selected)
    """

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    importances = model.feature_importances_

    if importances is None or len(importances) == 0:
        raise RuntimeError("XGBoost returned no feature importances. Cannot rank features.")

    indices = np.argsort(importances)[::-1]

    topk_idx = indices[:k]
    X_train_sel = X_train[:, topk_idx]
    X_test_sel = X_test[:, topk_idx]

    return X_train_sel, X_test_sel