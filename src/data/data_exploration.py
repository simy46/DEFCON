"""
Script for initial data exploration.
Used once to understand the dataset structure before building the pipeline.
Not imported in main.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.consts import METADATA_TRAIN_PATH, TRAIN_NPZ_PATH, X_TRAIN_KEY, Y_TRAIN_KEY


def explore_data():
    """
    Load metadata CSV and NPZ files.
    
    Returns:
        X : np.ndarray
        metadata : pd.DataFrame
        y : np.ndarray
    """
    metadata = pd.read_csv(METADATA_TRAIN_PATH)
    data = np.load(TRAIN_NPZ_PATH)

    X = data[X_TRAIN_KEY]
    y = data[Y_TRAIN_KEY]

    print(f"Data loaded:")
    print(f"  X shape        : {X.shape}")
    print(f"  y shape        : {y.shape}")
    print(f"  metadata shape : {metadata.shape}")

    return X, metadata, y


def analyze_metadata(metadata, ignore_cols=None, max_display=20):
    """
    Display unique values for each metadata column.
    """
    if ignore_cols is None:
        ignore_cols = ["Unnamed: 0", "ID", "Create date"]

    print("\nExploring metadata columns:\n")

    for col in metadata.columns:
        if col not in ignore_cols:
            unique_vals = metadata[col].dropna().unique()
            n = len(unique_vals)

            print(f"> {col}: {n} unique values")

            if n <= max_display:
                print("  Values:", unique_vals)
            else:
                print("  Values:", unique_vals[:max_display], "...")

            print("-" * 60)


def visualize_y(y, title="Distribution of target y"):
    """
    Show histogram and summary stats for the target vector.
    """
    unique, counts = np.unique(y, return_counts=True)

    print("\nLabel distribution:")
    for val, count in zip(unique, counts):
        print(f"  y = {val}: {count}")

    plt.figure(figsize=(5, 3))
    plt.bar(unique.astype(str), counts)
    plt.title(title)
    plt.xlabel("class")
    plt.ylabel("count")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()



## ENTRY POINT
X, meta, y = explore_data()

analyze_metadata(meta)
visualize_y(y)

print("\nCheck for invalid values:")
print("  NaN in X:", np.isnan(X).any())
print("  Inf in X:", np.isinf(X).any())
