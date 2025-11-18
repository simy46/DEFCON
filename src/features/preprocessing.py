import logging
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from config.consts import LOG_FILE_PREFIX

from .normalization import normalize_data
from .feature_selection import (
    filter_low_variance_features,
    select_first_features,
    select_k_best_features,
    select_xgboost_k_features
)
from .pca import apply_pca
from .encode_metadata import encode_metadata

logger = logging.getLogger(LOG_FILE_PREFIX)


def apply_preprocessing(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    metadata_train: DataFrame,
    metadata_test: DataFrame,
    cfg: dict
) -> tuple[NDArray, NDArray]:
    """
    Apply preprocessing steps based on YAML config.
    """
    pp = cfg["preprocessing"]

    # attributes
    ohe = pp["one_hot_encode"]
    select_xgboost_k = pp["select_xgboost_k"]
    select_best_k = pp["select_k_best"]
    select_first_k = pp["select_first_k"]
    pca = pp["pca"]
    normalize = pp["normalize"]
    variance_threshold = pp["variance_threshold"]

    logger.info(f"Initial shapes: X_train={X_train.shape}, X_test={X_test.shape}")


    # ---------------------------------------------
    # Select K Best Based On xgBoost (supervised)
    # ---------------------------------------------
    if select_xgboost_k["enabled"]:
        logger.info(f"Selecting top {k} most informative features via XGBoost...")
        select_xgboost_k_features(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            k=select_xgboost_k["k"],
        )
        logger.info(f"After XGBoost K select: X_train={X_train.shape}, X_test={X_test.shape}")


    # ---------------------------------------------
    # Select K Best (supervised)
    # ---------------------------------------------
    if select_best_k["enabled"]:
        k = select_best_k["k"]
        logger.info(f"Selecting top {k} most informative features ...")
        X_train, X_test = select_k_best_features(X_train, y_train, X_test, k)
        logger.info(f"After SelectKBest: X_train={X_train.shape}, X_test={X_test.shape}")


    # ---------------------------------------------
    # First-k feature selection (optional)
    # ---------------------------------------------
    if select_first_k["enabled"]:
        k = select_first_k["k"]
        logger.info(f"Selecting first {k} features ...")
        X_train, X_test = select_first_features(X_train, X_test, k)
        logger.info(f"After first-k selection: X_train={X_train.shape}, X_test={X_test.shape}")


    # ---------------------------------------------
    # Normalization
    # ---------------------------------------------
    if normalize:
        with_mean = pp.get("normalize_with_mean", False)
        logger.info(f"Applying StandardScaler(with_mean={with_mean}) ...")
        X_train, X_test = normalize_data(X_train, X_test, with_mean)
        logger.info(f"After normalization: X_train={X_train.shape}, X_test={X_test.shape}")

    
    # ---------------------------------------------
    # Variance threshold
    # ---------------------------------------------
    if variance_threshold["enabled"]:
        threshold = variance_threshold["threshold"]
        logger.info(f"Applying VarianceThreshold(threshold={threshold}) ...")
        X_train, X_test = filter_low_variance_features(X_train, X_test, threshold)
        logger.info(f"After VarianceThreshold: X_train={X_train.shape}, X_test={X_test.shape}")


    # ---------------------------------------------
    # PCA
    # ---------------------------------------------
    if pca["enabled"]:
        n_components = pca["n_components"]
        svd_solver = pca.get("svd_solver", "randomized")
        logger.info(f"Applying PCA(n_components={n_components}) ...")
        X_train, X_test = apply_pca(X_train, X_test, n_components, svd_solver)
        logger.info(f"After PCA: X_train={X_train.shape}, X_test={X_test.shape}")


        # ---------------------------------------------
    # One-Hot Encoding (metadata)
    # ---------------------------------------------
    if ohe["enabled"]:
        columns = ohe["columns"]
        logger.info(f"Applying One-Hot Encoding on metadata columns: {columns} ...")
        
        train_meta_enc, test_meta_enc = encode_metadata(
            metadata_train,
            metadata_test,
            columns
        )

        logger.info(f"Metadata encoded: train_meta={train_meta_enc.shape}, test_meta={test_meta_enc.shape}")

        X_train = np.hstack([X_train, train_meta_enc])
        X_test = np.hstack([X_test, test_meta_enc])

        logger.info(f"After metadata concat: X_train={X_train.shape}, X_test={X_test.shape}")

    logger.info("Preprocessing complete.")
    return X_train, X_test
