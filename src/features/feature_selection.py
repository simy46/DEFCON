from numpy.typing import NDArray
from sklearn.feature_selection import VarianceThreshold

def filter_low_variance_features(
    X_train: NDArray,
    X_test: NDArray,
    threshold: float
) -> tuple[NDArray, NDArray]:
    """
    Remove low-variance features.

    Args:
        threshold (float): Minimum variance required.

    Returns:
        tuple[NDArray, NDArray]: Filtered train and test matrices.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel


def select_first_features(
    X_train: NDArray,
    X_test: NDArray,
    n_features: int
) -> tuple[NDArray, NDArray]:
    """
    Select the first n_features columns.

    Args:
        n_features (int): Number of columns to keep.

    Returns:
        tuple[NDArray, NDArray]: Reduced train and test matrices.
    """
    return X_train[:, :n_features], X_test[:, :n_features]
