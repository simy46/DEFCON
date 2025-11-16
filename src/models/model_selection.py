from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # type: ignore
from config.consts import LOGISTIC_REGRESSION, SVM, RANDOM_FOREST, XGBOOST

def build_model(model_cfg):
    name = model_cfg["name"]

    if name == LOGISTIC_REGRESSION:
        return LogisticRegression(
            solver=model_cfg["solver"],
            penalty=model_cfg["penalty"],
            C=model_cfg["C"],
            max_iter=model_cfg["max_iter"],
            n_jobs=model_cfg["n_jobs"],
            random_state=model_cfg["random_state"]
        )

    if name == SVM:
        return SVC(
            C=model_cfg["C"],
            kernel=model_cfg["kernel"],
            probability=True
        )

    if name == RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            min_samples_split=model_cfg["min_samples_split"],
            min_samples_leaf=model_cfg["min_samples_leaf"],
            class_weight=model_cfg["class_weight"],
            n_jobs=model_cfg["n_jobs"]
        )

    if name == XGBOOST:
        return XGBClassifier(
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            learning_rate=model_cfg["learning_rate"],
            subsample=model_cfg["subsample"],
            colsample_bytree=model_cfg["colsample_bytree"],
            eval_metric=model_cfg["eval_metric"]
        )

    raise ValueError(f"Unknown model name: {name}")
