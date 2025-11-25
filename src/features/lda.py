from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np


def apply_lda(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    mode: str = "classic",
    n_components: int | None = None,
    shrinkage_value: float | None = None,
    kernel: str | None = None,
    gamma: float | None = None,
    degree: int = 3,
    coef0: float = 1.0
) -> tuple[NDArray, NDArray]:
    """
    Unified LDA function supporting:
    - Classic LDA
    - Regularized LDA
    - Shrinkage LDA
    - Kernel LDA

    Supports multidimensional output (up to C-1).
    """

    # ------------------------------------------------------------
    # Compute max possible dimensions (C - 1)
    # ------------------------------------------------------------
    classes = np.unique(y_train)
    max_dim = len(classes) - 1

    if max_dim < 1:
        raise ValueError("LDA requires at least 2 classes.")

    if n_components is None:
        n_components = max_dim
    else:
        n_components = min(n_components, max_dim)

    # ------------------------------------------------------------
    # CLASSIC LDA
    # ------------------------------------------------------------
    if mode == "classic":
        lda = LDA(
            solver="svd",
            n_components=n_components
        )
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)
        return X_train_lda, X_test_lda

    # ------------------------------------------------------------
    # REGULARIZED LDA
    # ------------------------------------------------------------
    elif mode == "regularized":
        if shrinkage_value is None:
            raise ValueError("shrinkage_value must be provided for regularized LDA")

        lda = LDA(
            solver="eigen",
            shrinkage=shrinkage_value,
            n_components=n_components
        )
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)
        return X_train_lda, X_test_lda

    # ------------------------------------------------------------
    # SHRINKAGE LDA (AUTO)
    # ------------------------------------------------------------
    elif mode == "shrinkage":
        lda = LDA(
            solver="eigen",
            shrinkage="auto",
            n_components=n_components
        )
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)
        return X_train_lda, X_test_lda

    # ------------------------------------------------------------
    # KERNEL LDA
    # ------------------------------------------------------------
    elif mode == "kernel":
        if kernel is None:
            raise ValueError("kernel must be provided for Kernel LDA")

        from sklearn.metrics.pairwise import (
            linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel
        )

        # Compute kernel matrices
        if kernel == "linear":
            K_train = linear_kernel(X_train)
            K_test = linear_kernel(X_test, X_train)

        elif kernel == "poly":
            K_train = polynomial_kernel(X_train, degree=degree, coef0=coef0)
            K_test = polynomial_kernel(X_test, X_train, degree=degree, coef0=coef0)

        elif kernel == "rbf":
            if gamma is None:
                raise ValueError("gamma must be provided for RBF kernel")
            K_train = rbf_kernel(X_train, gamma=gamma)
            K_test = rbf_kernel(X_test, X_train, gamma=gamma)

        elif kernel == "sigmoid":
            K_train = sigmoid_kernel(X_train, gamma=gamma, coef0=coef0)
            K_test = sigmoid_kernel(X_test, X_train, gamma=gamma, coef0=coef0)

        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

        # Apply classical LDA in kernel space
        lda = LDA(
            solver="svd",
            n_components=n_components
        )
        X_train_lda = lda.fit_transform(K_train, y_train)
        X_test_lda = lda.transform(K_test)
        return X_train_lda, X_test_lda

    # ------------------------------------------------------------
    # INVALID MODE
    # ------------------------------------------------------------
    else:
        raise ValueError(f"Unknown LDA mode '{mode}'")
