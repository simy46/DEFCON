import numpy as np
import time
from consts.consts import METADATA_TRAIN_PATH, TRAIN_NPZ_PATH
import xgboost as xgb
import optuna
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from Discovering import explore_data  # ton module existant
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

def optuna_search_random_forest(X, y, n_trials=60):
    import optuna

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 5, 60)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        model = RandomForestClassifier(
            class_weight="balanced",
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for tr_idx, val_idx in cv.split(X, y):
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], preds))

        return np.mean(scores)

    print("🚀 Optuna Random Forest: optimisation...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("🏆 Best parameters:", study.best_params)
    print("💯 Best F1-score:", study.best_value)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
    }


def optuna_search_logistic_regression(X, y, n_trials=60):

    def objective(trial):
        C = trial.suggest_loguniform("C", 1e-4, 1e2)
        solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "newton-cg", "saga"])
        max_iter = trial.suggest_int("max_iter", 200, 4000)

        model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            penalty="l2",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for tr_idx, val_idx in cv.split(X, y):
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], preds))

        return np.mean(scores)

    print("🚀 Optuna Logistic Regression: optimisation...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("🏆 Best parameters:", study.best_params)
    print("💯 Best F1-score:", study.best_value)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
    }

def optuna_search_svm(X, y, n_trials=60):

    def objective(trial):
        C = trial.suggest_loguniform("C", 1e-3, 1e2)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight="balanced",
            probability=True,
            random_state=42
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for tr_idx, val_idx in cv.split(X, y):
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], preds))

        return np.mean(scores)

    print("🚀 Optuna SVM: optimisation...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("🏆 Best parameters:", study.best_params)
    print("💯 Best F1-score:", study.best_value)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
    }

# -------------------------------
# LIGHTGBM
# -------------------------------
def optuna_search_lightgbm(X, y, n_trials=100):

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 12),
            "objective": "binary",
            "random_state": 42,
            "n_jobs": -1
        }

        model = lgb.LGBMClassifier(**params)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in cv.split(X, y):
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], preds))
        return np.mean(scores)

    print("🚀 Optuna LightGBM: optimisation...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("🏆 Best parameters:", study.best_params)
    print("💯 Best F1-score:", study.best_value)
    return {"best_params": study.best_params, "best_score": study.best_value}


# -------------------------------
# CATBOOST
# -------------------------------
def optuna_search_catboost(X, y, n_trials=100):

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 12),
            "random_state": 42,
            "verbose": 0,
            "thread_count": -1
        }

        model = CatBoostClassifier(**params)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in cv.split(X, y):
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], preds))
        return np.mean(scores)

    print("🚀 Optuna CatBoost: optimisation...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("🏆 Best parameters:", study.best_params)
    print("💯 Best F1-score:", study.best_value)
    return {"best_params": study.best_params, "best_score": study.best_value}


# -------------------------------
# MLP (Multi-layer Perceptron)
# -------------------------------
def optuna_search_mlp(X, y, n_trials=60):

    def objective(trial):
        hidden_layer_sizes = tuple(
            trial.suggest_int("n_units_l{}".format(i), 10, 200)
            for i in range(trial.suggest_int("n_layers", 1, 3))
        )
        activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
        solver = trial.suggest_categorical("solver", ["adam", "sgd"])
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
        max_iter = 1000

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=42
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in cv.split(X, y):
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], preds))
        return np.mean(scores)

    print("🚀 Optuna MLP: optimisation...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("🏆 Best parameters:", study.best_params)
    print("💯 Best F1-score:", study.best_value)
    return {"best_params": study.best_params, "best_score": study.best_value}


def optuna_search_xgboost(X, y, n_trials=300):
    import optuna
    from xgboost import XGBClassifier

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 12),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
        }

        model = XGBClassifier(**params)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for tr_idx, val_idx in cv.split(X, y):
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], preds))

        return np.mean(scores)

    print("🚀 Optuna XGBoost: optimisation...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("🏆 Best parameters:", study.best_params)
    print("💯 Best F1-score:", study.best_value)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
    }


if __name__ == "__main__":
    total_start = time.time()

    # Charger les données PCA
    npz_path_train = "train_pca_20_components.npz"

    # Charger via ta fonction existante
    X_train, X_metadata, _ = explore_data(METADATA_TRAIN_PATH, npz_path_train, ['X_pca'])

    y_train = np.load(TRAIN_NPZ_PATH)['y_train']
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
    #results = optuna_search_random_forest(X_tr, y_tr)
    #results = optuna_search_logistic_regression(X_tr, y_tr)
    #results = optuna_search_svm(X_tr, y_tr)
    #results = optuna_search_xgboost(X_tr, y_tr)
    #results = optuna_search_lightgbm(X_tr, y_tr)
    #results = optuna_search_catboost(X_tr, y_tr)
    #results = optuna_search_mlp(X_tr, y_tr)
    
    total_end = time.time()
    print(f"\n⏱️ Temps total du script : {(total_end - total_start)/60:.1f} minutes")
