from numpy.typing import NDArray
from sklearn.decomposition import PCA

def apply_pca(
    X_train: NDArray,
    X_test: NDArray,
    n_components: int,
    svd_solver: str
) -> tuple[NDArray, NDArray]:
    """
    Apply PCA dimensionality reduction.

    Args:
        n_components (int): Number of PCA dimensions.

    Returns:
        tuple[NDArray, NDArray]: PCA-transformed train and test matrices.
    """
    pca = PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        random_state=42
    )
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca
