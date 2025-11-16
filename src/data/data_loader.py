import os
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from src.utils.gzip_utils import gunzip_file
from config.consts import METADATA_TEST_PATH, METADATA_TRAIN_PATH, TEST_GZ_PATH, TEST_NPZ_PATH, TRAIN_GZ_PATH, TRAIN_NPZ_PATH, X_TEST_KEY, X_TRAIN_KEY, Y_TRAIN_KEY


def _ensure_npz(npz_path: str, gz_path: str) -> None:
    """
    Ensure NPZ exists; otherwise extract from .gz.

    Args:
        npz_path (str): Expected path of the .npz file.
        gz_path (str): Path of the .npz.gz file used to restore the NPZ.

    Returns:
        None
    """
    if os.path.exists(npz_path):
        print(f"[INFO] Found NPZ file: {npz_path}")
        return

    print(f"[WARNING] NPZ file missing: {npz_path}")
    print("[INFO] Decompressing GZIP...")

    if not os.path.exists(gz_path):
        raise FileNotFoundError(
            f"Missing GZIP file: {gz_path}\n"
            f"Cannot restore {npz_path}"
        )

    gunzip_file(gz_path, npz_path)

    if not os.path.exists(npz_path):
        raise RuntimeError(
            f"[ERROR] Extraction failed: {npz_path} still does not exist."
        )

    print(f"[INFO] Successfully extracted: {npz_path}")


def load_train() -> tuple[NDArray, DataFrame, NDArray]:
    """
    Load training NPZ (X and y) and metadata using the config file.

    Args:
        cfg (dict): YAML configuration dict containing paths and keys.

    Returns:
        tuple:
            X (np.ndarray): Feature matrix for training data.
            metadata (pd.DataFrame): Training metadata.
            y (np.ndarray): Label vector.
    """

    _ensure_npz(TRAIN_NPZ_PATH, TRAIN_GZ_PATH)

    data = np.load(TRAIN_NPZ_PATH, allow_pickle=True)

    X: NDArray = data[X_TRAIN_KEY]
    y: NDArray = data[Y_TRAIN_KEY]

    metadata: DataFrame = pd.read_csv(METADATA_TRAIN_PATH)

    return X, metadata, y


def load_test() -> tuple[NDArray, DataFrame]:
    """
    Load test NPZ and metadata using the config file.

    Args:
        cfg (dict): YAML configuration dict containing paths and keys.

    Returns:
        tuple:
            X_test (np.ndarray): Feature matrix for the test data.
            metadata_test (pd.DataFrame): Test metadata.
    """

    _ensure_npz(TEST_NPZ_PATH, TEST_GZ_PATH)

    data = np.load(TEST_NPZ_PATH, allow_pickle=True)

    X_test: NDArray = data[X_TEST_KEY]
    metadata_test: DataFrame = pd.read_csv(METADATA_TEST_PATH)

    return X_test, metadata_test
