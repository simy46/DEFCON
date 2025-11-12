import numpy as np
import pandas as pd

# Charger le fichier .npz
data = np.load("train.npz", allow_pickle=True)

# Vérifier que la clé existe
if 'y_train' in data.files:
    y_train = data['y_train']  # Extraire le tableau
    # Convertir en DataFrame et sauvegarder en CSV
    pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
    print("✅ y_train.csv a été créé avec succès !")
else:
    print("❌ La clé 'y_train' n'existe pas dans le fichier .npz")