import logging
from numpy.typing import NDArray

from .normalization import normalize_data
from .feature_selection import (
    filter_low_variance_features,
    select_first_features
)
from .pca import apply_pca

logger = logging.getLogger(__name__)


def apply_preprocessing(
    X_train: NDArray,
    X_test: NDArray,
    cfg: dict
) -> tuple[NDArray, NDArray]:
    """
    Apply preprocessing steps based on YAML config.
    """
    pp = cfg["preprocessing"]

    # attributes
    variance_threshold = pp["variance_threshold"]
    pca = pp["pca"]
    select_first_k = pp["select_first_k"]
    normalize = pp["normalize"]

    logger.info(f"Initial shapes: X_train={X_train.shape}, X_test={X_test.shape}")


    # ---------------------------------------------
    # First-k feature selection
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

    logger.info("Preprocessing complete.")
    return X_train, X_test
