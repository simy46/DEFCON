from sklearn.calibration import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # type: ignore
from config.consts import LOGISTIC_REGRESSION, LINEAR_SVM, RANDOM_FOREST, XGBOOST

def build_model(model_cfg):
    name = model_cfg["name"]

    if name == LOGISTIC_REGRESSION:
        return LogisticRegression(
            solver=model_cfg["solver"],
            penalty=model_cfg["penalty"],
            C=model_cfg["C"],
            max_iter=model_cfg["max_iter"],
            n_jobs=model_cfg["n_jobs"],
            random_state=model_cfg["random_state"],
        )

    if name == LINEAR_SVM:
        return LinearSVC(
            C=model_cfg["C"],
            loss=model_cfg["loss"],
            max_iter=model_cfg["max_iter"],
            random_state=model_cfg["random_state"],
        )

    if name == RANDOM_FOREST:
        return RandomForestClassifier(
            n_jobs=model_cfg["n_jobs"],
            n_estimators=model_cfg["n_estimators"],
            min_samples_split=model_cfg["min_samples_split"],
            min_samples_leaf=model_cfg["min_samples_leaf"],
            max_features=model_cfg["max_features"],
            max_depth=model_cfg["max_depth"],
            criterion=model_cfg["criterion"],
            bootstrap=model_cfg["bootstrap"],
            class_weight=model_cfg["class_weight"],
            random_state=model_cfg["random_state"],
        )

    if name == XGBOOST:
        return XGBClassifier(
        booster=model_cfg.get("booster", "gbtree"),
        eta=model_cfg.get("eta", 0.1),
        max_depth=model_cfg["max_depth"],
        min_child_weight=model_cfg.get("min_child_weight", 1),
        subsample=model_cfg["subsample"],
        colsample_bytree=model_cfg["colsample_bytree"],
        reg_lambda=model_cfg.get("lambda", 1.0),
        reg_alpha=model_cfg.get("alpha", 0.0),
        n_estimators=model_cfg["n_estimators"],
        tree_method=model_cfg.get("tree_method", "hist"),
        max_bin=model_cfg.get("max_bin", 256),
        n_jobs=model_cfg.get("n_jobs", -1),
        random_state=model_cfg.get("random_state", 42),
        eval_metric=model_cfg.get("eval_metric", "logloss")
    )

    raise ValueError(f"Unknown model name: {name}")
