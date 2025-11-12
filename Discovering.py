import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def explore_data(csv_path, npz_path, keys):
    """
    Charge les fichiers CSV et NPZ et retourne les tableaux utiles.
    
    Returns:
        X : np.ndarray — données d'entrée (features)
        X_metadata : pd.DataFrame — métadonnées descriptives
        y : np.ndarray — labels
    """
    # Charger les métadonnées
    X_metadata = pd.read_csv(csv_path)

    # Charger le fichier .npz
    data = np.load(npz_path)

    # Extraire les clés
    X = data[keys[0]]
    y = data[keys[1]] if len(keys) > 1 else None

    print(f"✅ Données chargées : X = {X.shape}, y = {y.shape if y is not None else ''}, meta = {X_metadata.shape}")

    return X, X_metadata, y


def analyze_metadata(X_metadata, ignore_cols=None, max_display=20):
    """
    Explore les métadonnées et affiche toutes les valeurs possibles
    pour les colonnes catégorielles.
    """
    if ignore_cols is None:
        ignore_cols = ["Unnamed: 0", "ID", "Create date"]

    print("🔍 Exploration des métadonnées :\n")

    for col in X_metadata.columns:
        if col not in ignore_cols:
            unique_vals = X_metadata[col].dropna().unique()
            n_unique = len(unique_vals)

            print(f"🧩 Colonne : {col}")
            print(f"Nombre de valeurs uniques : {n_unique}")

            if n_unique <= max_display:
                print("→", unique_vals)
            else:
                print("→", unique_vals[:max_display], "... (troncature)")
            print("-" * 60)
    print("✅ Exploration des métadonnées terminée.")

def visualize_y(y, title="Distribution des valeurs de y"):
    """
    Affiche un résumé et une visualisation du vecteur y (labels).
    """
    print("📏 Dimensions de y :", y.shape)
    print("🔢 Type :", y.dtype)

    # Si y contient des valeurs catégorielles (ex: 0/1 ou labels texte)
    unique, counts = np.unique(y, return_counts=True)
    print("\nValeurs uniques et fréquences :")
    for val, c in zip(unique, counts):
        print(f"  - {val} : {c}")

    # Si le nombre de valeurs uniques est petit, on fait un histogramme discret
    plt.figure(figsize=(7, 4))
    plt.bar(unique.astype(str), counts)
    plt.xlabel("Valeur de y")
    plt.ylabel("Nombre d'occurrences")
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":

    # Charger les données train
    X, X_meta, y = explore_data("metadata_train.csv", "train.npz", ['X_train', 'y_train'])

    # Explorer les métadonnées
    analyze_metadata(X_meta)
    
    # Visualiser y
    visualize_y(y)

    # Vérifier la présence de NaN ou Inf dans X
    print("Présence de NaN dans X :", np.isnan(X).any())
    print("Présence d'Inf dans X :", np.isinf(X).any())
