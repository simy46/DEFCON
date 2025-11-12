from sklearn.model_selection import train_test_split
import time
from Discovering import explore_data
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import numpy as np
import pandas as pd
from Finding_model import split_data

if __name__ == "__main__":

    # Total time tracking
    total_start_time = time.time()

    # Chemins des fichiers train
    csv_path_train = "metadata_train.csv"
    npz_path_train = "train_pca.npz"

    # Charger les données
    X_train, X_metadata, y_train = explore_data(csv_path_train, npz_path_train, ['X_pca', 'y'])

    # Diviser les données
    X_tr, X_val, y_tr, y_val = split_data(X_train, y_train, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=300, bootstrap= False, max_depth = 10, min_samples_leaf= 4, min_samples_split = 2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    print(classification_report(y_val, y_pred))

    X_test, X_metadata_test, y_test = explore_data("metadata_test.csv", "test_pca.npz", ['X_pca'])

    print(X_metadata_test)

    y_test_pred = model.predict(X_test)
    
    # Sauvegarder les prédictions dans un fichier CSV
    submission_df = pd.DataFrame({
        "id": X_metadata_test["ID"],  # 0,1,2,...,1091
        "label": y_test_pred            # tes prédictions
    })
    
    submission_df.to_csv("submission.csv", index=False)

    print(submission_df.shape)  # Doit afficher (1092, 1)
    print(submission_df.head())
    print("✅ Prédictions sauvegardées dans test_predictions.csv")

