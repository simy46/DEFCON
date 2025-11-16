import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def explore_data(csv_path, npz_path, keys):
    """
    Load CSV and NPZ files and return useful arrays.
    
    Returns:
        X : np.ndarray — input features
        X_metadata : pd.DataFrame — descriptive metadata
        y : np.ndarray — labels
    """
    X_metadata = pd.read_csv(csv_path)
    data = np.load(npz_path)

    X = data[keys[0]]
    y = data[keys[1]] if len(keys) > 1 else None

    print(f"Data loaded: X = {X.shape}, y = {y.shape if y is not None else ''}, metadata = {X_metadata.shape}")

    return X, X_metadata, y


def analyze_metadata(X_metadata, ignore_cols=None, max_display=20):
    """
    Explore metadata and display all possible values for categorical columns.
    """
    if ignore_cols is None:
        ignore_cols = ["Unnamed: 0", "ID", "Create date"]

    print("Exploring metadata:\n")

    for col in X_metadata.columns:
        if col not in ignore_cols:
            unique_vals = X_metadata[col].dropna().unique()
            n_unique = len(unique_vals)

            print(f"Column: {col}")
            print(f"Number of unique values: {n_unique}")

            if n_unique <= max_display:
                print("→", unique_vals)
            else:
                print("→", unique_vals[:max_display], "... (truncated)")
            print("-" * 60)
    print("Metadata exploration finished.")


def visualize_y(y, title="Distribution of y values"):
    """
    Display a summary and visualization of the label vector y.
    """
    print("y dimensions:", y.shape)
    print("Type:", y.dtype)

    # If y contains categorical values (e.g., 0/1 or text labels)
    unique, counts = np.unique(y, return_counts=True)
    print("\nUnique values and frequencies:")
    for val, c in zip(unique, counts):
        print(f"  - {val} : {c}")

    # If the number of unique values is small, make a discrete histogram
    plt.figure(figsize=(7, 4))
    plt.bar(unique.astype(str), counts)
    plt.xlabel("y value")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    X, X_meta, y = explore_data("metadata_train.csv", "train.npz", ['X_train', 'y_train'])
    analyze_metadata(X_meta)
    visualize_y(y)
    print("Presence of NaN in X:", np.isnan(X).any())
    print("Presence of Inf in X:", np.isinf(X).any())
