import logging
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from .model_selection import build_model
from config.consts import LOG_FILE_PREFIX


logger = logging.getLogger(LOG_FILE_PREFIX)

def train_model(X_train, y_train, cfg):
    """
    Train model using StratifiedKFold cross-validation.
    Refit the model on full training data at the end.
    """

    model_cfg = cfg["model"]
    random_state = model_cfg["random_state"]
    k_folds = model_cfg["cv_folds"]

    model = build_model(model_cfg)

    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    fold_f1 = []
    fold_f1_train = []

    logger.info(f"Starting {k_folds}-fold cross-validation...")

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        logger.info(f"--- Fold {fold_idx + 1}/{k_folds} ---")

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # Train
        model.fit(X_tr, y_tr)

        # Predictions
        y_pred = model.predict(X_val)
        y_pred_train = model.predict(X_tr)

        # Metrics
        f1 = f1_score(y_val, y_pred)
        f1_tr = f1_score(y_tr, y_pred_train)

        fold_f1.append(f1)
        fold_f1_train.append(f1_tr)

        logger.info(f"  Train F1: {f1_tr:.4f}")
        logger.info(f"  Valid F1: {f1:.4f}")
        logger.info(f"  Gap (train - val): {f1_tr - f1:.4f}")

    # Summary
    logger.info("Cross-validation complete.")
    logger.info(f"Mean F1 = {np.mean(fold_f1):.4f}")
    logger.info(f"Std F1  = {np.std(fold_f1):.4f}")

    # Refit final model
    logger.info("Refitting final model on full dataset...")
    model.fit(X_train, y_train)

    return model
