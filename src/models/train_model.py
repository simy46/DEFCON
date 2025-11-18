import logging
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from .model_selection import build_model
from config.consts import LOG_FILE_PREFIX


logger = logging.getLogger(LOG_FILE_PREFIX)

def train_model(X_train, y_train, cfg):
    """
    Train model using StratifiedKFold cross-validation.
    Logs both binary-F1 and macro-F1.
    Refit the final model on full dataset.
    """

    model_cfg = cfg["model"]
    random_state = model_cfg["random_state"]
    k_folds = model_cfg["cv_folds"]

    model = build_model(model_cfg)

    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    f1_binary_scores = []
    f1_macro_scores = []
    f1_binary_train_scores = []
    f1_macro_train_scores = []

    logger.info(f"Starting {k_folds}-fold cross-validation...")

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        logger.info(f"--- Fold {fold_idx + 1}/{k_folds} ---")

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)

        y_pred_val = model.predict(X_val)
        y_pred_tr = model.predict(X_tr)

        f1_bin = f1_score(y_val, y_pred_val, average="binary")
        f1_mac = f1_score(y_val, y_pred_val, average="macro")

        f1_bin_tr = f1_score(y_tr, y_pred_tr, average="binary")
        f1_mac_tr = f1_score(y_tr, y_pred_tr, average="macro")

        f1_binary_scores.append(f1_bin)
        f1_macro_scores.append(f1_mac)
        f1_binary_train_scores.append(f1_bin_tr)
        f1_macro_train_scores.append(f1_mac_tr)

        logger.info(f"  Train F1 (binary): {f1_bin_tr:.4f}")
        logger.info(f"  Train F1 (macro) : {f1_mac_tr:.4f}")
        logger.info(f"  Valid F1 (binary): {f1_bin:.4f}")
        logger.info(f"  Valid F1 (macro) : {f1_mac:.4f}")
        logger.info(f"  Gap (binary):     {f1_bin_tr - f1_bin:.4f}")
        logger.info(f"  Gap (macro):      {f1_mac_tr - f1_mac:.4f}")

    logger.info("Cross-validation complete.")
    logger.info(f"Mean F1 (binary) = {np.mean(f1_binary_scores):.4f}")
    logger.info(f"Std  F1 (binary) = {np.std(f1_binary_scores):.4f}")
    logger.info(f"Mean F1 (macro)  = {np.mean(f1_macro_scores):.4f}")
    logger.info(f"Std  F1 (macro)  = {np.std(f1_macro_scores):.4f}")

    logger.info("Refitting final model on full dataset...")
    model.fit(X_train, y_train)

    return model
