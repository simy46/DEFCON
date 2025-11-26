from typing import List, Tuple
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder


def encode_metadata(
    metadata_train: DataFrame,
    metadata_test: DataFrame,
    columns: List[str]
) -> Tuple[NDArray, NDArray]:
    """
    One-hot encode specified categorical metadata columns.
    Args:
        metadata_train (DataFrame):
            Metadata associated with training samples.
        metadata_test (DataFrame):
            Metadata associated with test samples.
        columns (List[str]):
            List of column names (categorical) to encode.
    Returns:
        Tuple[NDArray, NDArray]:
            - Encoded training metadata as dense NumPy array.
            - Encoded test metadata as dense NumPy array.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_encoded: NDArray = encoder.fit_transform(metadata_train[columns])
    test_encoded: NDArray = encoder.transform(metadata_test[columns])

    return train_encoded, test_encoded