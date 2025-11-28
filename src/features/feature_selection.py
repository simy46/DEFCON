from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from xgboost import XGBClassifier
import os


# ============================================================
# FEATURE SELECTION CONSTANTS
# ============================================================

VARIANCE_AXIS = 0
TOP_VARIANCE_K = 10000
K_BEST_CHI2_K = 20000
XGB_SELECTOR_K = 200
FIRST_K = 200000

XGB_SELECTOR_N_EST = 60
XGB_SELECTOR_MAX_DEPTH = 4
XGB_SELECTOR_LEARNING_RATE = 0.01
XGB_SELECTOR_SUBSAMPLE = 0.45
XGB_SELECTOR_COLSAMPLE = 0.35
XGB_OBJECTIVE = "binary:logistic"
XGB_EVAL_METRIC = "logloss"
XGB_TREE_METHOD = "hist"
XGB_N_JOBS = -1
RANDOM_STATE = 42


def filter_low_variance_features(
    X_train: NDArray,
    X_test: NDArray,
    threshold: float,
    data_dir: str = "../../data/"
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
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, f"X_train_var_{threshold}.npy")
    test_path = os.path.join(data_dir, f"X_test_var_{threshold}.npy")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Already found a thresholded dataset!")
        X_train_sel = np.load(train_path)
        X_test_sel = np.load(test_path)
        return X_train_sel, X_test_sel
    
    print("Applying the threshold")
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel: NDArray = selector.fit_transform(X_train)
    X_test_sel: NDArray = selector.transform(X_test)
    np.save(train_path, X_train_sel)
    np.save(test_path, X_test_sel)
    print("saving thresholded data")
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
    Args:
        X_train (NDArray):
            Training feature matrix.
        y_train (NDArray):
            Training labels.
        X_test (NDArray):
            Test feature matrix.
        k (int):
            Number of best features to keep.
    Returns:
        Tuple[NDArray, NDArray]:
            Tuple containing (X_train_selected, X_test_selected), where each array
            contains only the k most informative features.
    """
    assert (X_train >= 0).all()
    selector = SelectKBest(chi2, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel


def select_xgboost_k_features(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    k: int
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
        n_estimators=XGB_SELECTOR_N_EST,
        max_depth=XGB_SELECTOR_MAX_DEPTH,
        learning_rate=XGB_SELECTOR_LEARNING_RATE,
        subsample=XGB_SELECTOR_SUBSAMPLE,
        colsample_bytree=XGB_SELECTOR_COLSAMPLE,
        objective=XGB_OBJECTIVE,
        eval_metric=XGB_EVAL_METRIC,
        tree_method=XGB_TREE_METHOD,
        n_jobs=XGB_N_JOBS,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    importances = model.feature_importances_
    if importances is None or len(importances) == 0:
        raise RuntimeError("XGBoost returned no feature importances. Cannot rank features.")
    indices = np.argsort(importances)[::-1]
    topk_idx = indices[:k]
    return X_train[:, topk_idx], X_test[:, topk_idx]



def select_top_variance_features(
    X_train: NDArray,
    X_test: NDArray,
    k: int
) -> Tuple[NDArray, NDArray]:
    """
    Select the top-k features with the highest variance in X_train.

    Useful for high-dimensional genomic data where most features are constant.

    Args:
        X_train (NDArray):
            Training matrix (n_samples, n_features).
        X_test (NDArray):
            Test matrix with same number/order of features.
        k (int):
            Number of highest-variance features to keep.

    Returns:
        Tuple[NDArray, NDArray]:
            Tuple containing (X_train_selected, X_test_selected), where each array
            contains only the k highest-variance features.
    """

    if k > X_train.shape[1]:
        raise ValueError(f"k={k} > number of features={X_train.shape[1]}")

    variances = X_train.var(axis=VARIANCE_AXIS)
    topk_idx = np.argsort(variances)[::-1][:k]

    return X_train[:, topk_idx], X_test[:, topk_idx]