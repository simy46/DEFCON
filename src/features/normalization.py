from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

def normalize_data(
    X_train: NDArray,
    X_test: NDArray,
    with_mean: bool = False
) -> tuple[NDArray, NDArray]:
    """
    Normalize train and test matrices using StandardScaler.

    Args:
        X_train (NDArray): Training feature matrix.
        X_test (NDArray): Test feature matrix.
        with_mean (bool): Whether to center the data (False recommended for sparse).

    Returns:
        tuple[NDArray, NDArray]: Normalized train and test matrices.
    """
    scaler = StandardScaler(with_mean=with_mean)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
