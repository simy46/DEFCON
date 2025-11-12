from sklearn.model_selection import train_test_split
import time
from Discovering import explore_data
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import numpy as np


def split_data(X_pca, y, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles d'entraînement et de test.

    Args:
        X_pca (np.ndarray): données d'entrée réduites par PCA
        y (np.ndarray): labels correspondants
        test_size (float): proportion des données à utiliser pour le test
        random_state (int): graine pour la reproductibilité

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"✅ Données divisées :")
    print(f"   → X_train : {X_train.shape}, y_train : {y_train.shape}")
    print(f"   → X_test  : {X_test.shape}, y_test  : {y_test.shape}")
    return X_train, X_test, y_train, y_test

def test_models(models, X_train, y_train, X_test, y_test):
    """
    Teste plusieurs modèles de classification et affiche leurs performances.

    Args:
        models (dict): dictionnaire de modèles à tester
        X_train (np.ndarray): données d'entraînement
        y_train (np.ndarray): labels d'entraînement
        X_test (np.ndarray): données de test
        y_test (np.ndarray): labels de test
    """

    # Évaluation
    for name, model in models.items():
        start_time = time.time()
        print(f"\n🔹 {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(classification_report(y_test, y_pred))

        elapsed_time = time.time() - start_time
        print(f"⏱️ Temps écoulé : {elapsed_time:.2f} secondes")



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

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
        "Linear SVM": SVC(kernel='linear', class_weight='balanced', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    }

    test_models(models, X_tr, y_tr, X_val, y_val)

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    print(f"⏱️ Temps total de traitement : {total_elapsed:.2f} secondes")

