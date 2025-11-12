import numpy as np
import time
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from Discovering import explore_data  # ton module existant

def grid_search_random_forest(X, y):
    """
    Effectue une recherche par grille (GridSearchCV) sur un RandomForestClassifier.
    
    Args:
        X (np.ndarray): données d'entrée (features)
        y (np.ndarray): labels cibles (binaire)
    
    Returns:
        dict: meilleur modèle, meilleurs paramètres et meilleurs scores
    """

    print("⚙️  Démarrage du Grid Search pour Random Forest...")
    start_time = time.time()

    # Paramètres à explorer
    param_grid = {
        'n_estimators': [50, 100, 300, 500],
        'max_depth': [None, 10, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
    }

    rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation stratifiée pour bien gérer le déséquilibre
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',  # métrique principale
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X, y)

    elapsed = time.time() - start_time
    print(f"\n✅ Grid Search terminé en {elapsed/60:.1f} minutes")
    print(f"🏆 Meilleurs paramètres : {grid_search.best_params_}")
    print(f"💯 Meilleur score f1 : {grid_search.best_score_:.4f}")

    return {
        "best_estimator": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
    }


if __name__ == "__main__":
    total_start = time.time()

    # Charger les données PCA
    csv_path_train = "metadata_train.csv"
    npz_path_train = "train_pca.npz"

    # Charger via ta fonction existante
    X_train, X_metadata, y_train = explore_data(csv_path_train, npz_path_train, ['X_pca', 'y'])

    # Split train/test pour évaluer ensuite le meilleur modèle
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # Lancer le grid search
    results = grid_search_random_forest(X_tr, y_tr)
    
    total_end = time.time()
    print(f"\n⏱️ Temps total du script : {(total_end - total_start)/60:.1f} minutes")
