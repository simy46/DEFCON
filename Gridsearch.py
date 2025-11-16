import numpy as np
import time
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import OneHotEncoder
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
        'bootstrap': [True],
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

def grid_search_logistic_regression(X, y):
    """
    Effectue une recherche par grille (GridSearchCV) sur un LogisticRegression.
    
    Args:
        X (np.ndarray): données d'entrée (features)
        y (np.ndarray): labels cibles (binaire)
    
    Returns:
        dict: meilleur modèle, meilleurs paramètres et meilleurs scores
    """

    print("⚙️  Démarrage du Grid Search pour Logistic Regression...")
    start_time = time.time()

    from sklearn.linear_model import LogisticRegression

    # Paramètres à explorer
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": [
        "liblinear",   # rapide sur petits datasets
        "lbfgs",       # stable
        "newton-cg",   # très bon pour L2, plus lent
        "saga"         # tolère les gros datasets + régularisation
    ],
        "penalty": ["l2"],
        "max_iter": [500, 1000, 2000, 4000]
    }

    model = LogisticRegression(
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    # Cross-validation stratifiée pour bien gérer le déséquilibre
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
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

def grid_search_svm(X, y):
    """
    Effectue une recherche par grille (GridSearchCV) sur un SVM (SVC).
    
    Args:
        X (np.ndarray): données d'entrée (features)
        y (np.ndarray): labels cibles

    Returns:
        dict: meilleur modèle, meilleurs paramètres et meilleurs scores
    """

    print("⚙️  Démarrage du Grid Search pour SVM...")
    start_time = time.time()

    # Paramètres à explorer
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
        "class_weight": ["balanced"]
    }

    model = SVC(
        probability=True,
        random_state=42
    )

    # Cross-validation stratifiée
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X, y)

    elapsed = time.time() - start_time
    print(f"\n✅ Grid Search SVM terminé en {elapsed/60:.1f} minutes")
    print(f"🏆 Meilleurs paramètres SVM : {grid_search.best_params_}")
    print(f"💯 Meilleur score f1 : {grid_search.best_score_:.4f}")

    return {
        "best_estimator": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
    }

def grid_search_xgboost(X, y):
    """
    Effectue une recherche par grille (GridSearchCV) sur XGBoostClassifier.
    
    Args:
        X (np.ndarray): données d'entrée (features)
        y (np.ndarray): labels cibles

    Returns:
        dict: meilleur modèle, meilleurs paramètres et meilleurs scores
    """

    print("⚙️  Démarrage du Grid Search pour XGBoost...")
    start_time = time.time()

    from xgboost import XGBClassifier

    # Paramètres à explorer
    param_grid = {
        "n_estimators": [50, 100, 300],
        "max_depth": [3, 5, 8],
        "learning_rate": [0.1],
        "subsample": [1.0],
        "colsample_bytree": [0.7, 1.0], 
        "gamma": [0, 1, 5],
        "scale_pos_weight": [1, 3, 6]   # très important pour dataset déséquilibré
    }

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"  # rapide pour gros dataset
    )

    # Cross-validation stratifiée
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X, y)

    elapsed = time.time() - start_time
    print(f"\n✅ Grid Search XGBoost terminé en {elapsed/60:.1f} minutes")
    print(f"🏆 Meilleurs paramètres XGBoost : {grid_search.best_params_}")
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
    npz_path_train = "train_pca_20_components.npz"
    y_train_path = "train.npz"

    # Charger via ta fonction existante
    X_train, X_metadata, _ = explore_data(csv_path_train, npz_path_train, ['X_pca'])

    y_train = np.load(y_train_path)['y_train']
     # X_metadata est un DataFrame (normalement)
    metadata = X_metadata.copy()

    # Colonnes catégorielles à inclure
    cat_cols = ["Isolation source", "Testing standard", "Isolation type", "Location" ]

    # Encoder les colonnes catégorielles
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_metadata = encoder.fit_transform(metadata[cat_cols])

    print(f"🔤 Encoded metadata shape: {encoded_metadata.shape}")

    # Concaténer PCA + metadata encodée
    X_extended = np.hstack([X_train, encoded_metadata])

    # Split train/test pour évaluer ensuite le meilleur modèle
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_extended, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # Lancer le grid search
    #results = grid_search_random_forest(X_tr, y_tr)
    #results = grid_search_logistic_regression(X_tr, y_tr)
    #results = grid_search_svm(X_tr, y_tr)
    #results = grid_search_xgboost(X_tr, y_tr)
    
    total_end = time.time()
    print(f"\n⏱️ Temps total du script : {(total_end - total_start)/60:.1f} minutes")
