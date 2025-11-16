import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from Discovering import explore_data
import time


def normalize_data(X):
    """
    Normalise les données X pour avoir une moyenne de 0 et un écart-type de 1.
    """
    start_time = time.time()  # Démarre le chronomètre

    scaler = StandardScaler(with_mean=False)  # avec_mean=False pour les grandes matrices
    X_scaled = scaler.fit_transform(X)

    print(f"✅ Données normalisées : {X.shape} -> {X_scaled.shape}")
    return X_scaled

def filter_low_variance_features(X, threshold=0.01):
    """
    Filtre les features de faible variance.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    print(f"✅ Features filtrées : {X.shape} -> {X_filtered.shape}")
    return X_filtered

def reduce_dimensions(X, n_components=300):
    """
    Réduit la dimensionnalité de X en utilisant PCA.
    """
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    X_reduced = pca.fit_transform(X)
    print(f"✅ Dimension réduite : {X.shape} -> {X_reduced.shape}")
    return X_reduced

def select_first_features(X, n_features):
    """
    Sélectionne les n premiers features (colonnes) du dataset X.

    Args:
        X (np.ndarray): tableau de données (n_samples, n_features_totales)
        n_features (int): nombre de features à garder (doit être <= nombre total de colonnes)

    Returns:
        np.ndarray: tableau réduit (n_samples, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("❌ X doit être un numpy.ndarray")

    n_total = X.shape[1]
    if n_features > n_total:
        raise ValueError(f"❌ n_features ({n_features}) > nombre total de features ({n_total})")

    X_reduced = X[:, :n_features]
    print(f"✅ X réduit à {X_reduced.shape[1]} features sur {n_total} ({100 * n_features / n_total:.2f} %)")
    return X_reduced

def save_pca_data(X_pca, y=None, output_prefix="train_pca"):
    """
    Sauvegarde les données réduites (PCA) et leurs labels dans un fichier .npz
    """
    output_path = f"{output_prefix}.npz"
    if y is not None:
        np.savez_compressed(output_path, X_pca=X_pca, y=y)
        print(f"✅ Données PCA sauvegardées sous : {output_path}")
        print(f"   → Dimensions sauvegardées : X_pca={X_pca.shape}, y={y.shape}")
    else:
        np.savez_compressed(output_path, X_pca=X_pca)
        print(f"✅ Données PCA (sans labels) sauvegardées sous : {output_path}")
        print(f"   → Dimensions sauvegardées : X_pca={X_pca.shape}")

if __name__ == "__main__":

    # Total time tracking
    total_start_time = time.time()

    # Chemins des fichiers train
    csv_path_train = "metadata_train.csv"
    npz_path_train = "train.npz"

    # Chemins des fichiers test
    csv_path_test = "metadata_test.csv"
    npz_path_test = "test.npz"

    # Charger les données
    X_train, X_metadata, y_train = explore_data(csv_path_train, npz_path_train, ['X_train', 'y_train'])
    X_test, X_metadata_test, y_test = explore_data(csv_path_test, npz_path_test, ['X_test'])

    # Réduire la dimensionnalité avec PCA
    start_time = time.time()  # Démarre le chronomètre

    X_reduced_train = reduce_dimensions(X_train, n_components=100)

    end_time = time.time()    # Arrête le chronomètre
    elapsed = end_time - start_time
    print(f"⏱️ Temps d'exécution : {elapsed:.2f} secondes")

    # Normaliser les données
    start_time = time.time()  # Démarre le chronomètre

    X_normalized_train = normalize_data(X_reduced_train)

    end_time = time.time()    # Arrête le chronomètre
    elapsed = end_time - start_time
    print(f"⏱️ Temps d'exécution : {elapsed:.2f} secondes")

    print("✅ Prétraitement terminé. Données prêtes pour l'entraînement.")

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time

    print(f"⏱️ Temps total de traitement : {total_elapsed:.2f} secondes")

    # Sauvegarder les données PCA (et normalisées)
    save_pca_data(X_normalized_train, None, output_prefix="train_pca_100_components")

    