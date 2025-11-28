from sklearn.calibration import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier 
from xgboost import XGBClassifier # type: ignore
from config.consts import LOGISTIC_REGRESSION, SVM, RANDOM_FOREST, XGBOOST, STACK

def build_model(model_cfg):
    name = model_cfg["name"]

    if name == LOGISTIC_REGRESSION:
        return LogisticRegression(
            solver=model_cfg["solver"],
            penalty=model_cfg["penalty"],
            class_weight='balanced',
            C=model_cfg["C"],
            l1_ratio=model_cfg["l1_ratio"],
            max_iter=model_cfg["max_iter"],
            n_jobs=model_cfg["n_jobs"],
            random_state=model_cfg["random_state"],
        )

    if name == SVM:
        return SVC(
            C=model_cfg["C"],
            kernel='rbf',
            gamma=model_cfg["gamma"],
            shrinking=False,
            probability=True,
            tol=model_cfg["tol"],
            class_weight='balanced',
            max_iter=model_cfg["max_iter"],
            break_ties=False,
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
        reg_lambda=model_cfg.get("reg_lambda", 1.0),
        reg_alpha=model_cfg.get("reg_alpha", 0.0),
        n_estimators=model_cfg["n_estimators"],
        tree_method=model_cfg.get("tree_method", "hist"),
        max_bin=model_cfg.get("max_bin", 256),
        n_jobs=model_cfg.get("n_jobs", -1),
        random_state=model_cfg.get("random_state", 42),
        eval_metric=model_cfg.get("eval_metric", "logloss")
    )

    if name == "stack":
        base_lr = build_model(model_cfg["lr"])
        base_svm = build_model(model_cfg["svm"])

        base_rf = RandomForestClassifier(
            n_estimators=model_cfg["rf"]["n_estimators"],
            min_samples_split=model_cfg["rf"]["min_samples_split"],
            min_samples_leaf=model_cfg["rf"]["min_samples_leaf"],
            max_features=model_cfg["rf"]["max_features"],
            max_depth=model_cfg["rf"]["max_depth"],
            criterion=model_cfg["rf"]["criterion"],
            bootstrap=model_cfg["rf"]["bootstrap"],
            class_weight=model_cfg["rf"]["class_weight"],
            random_state=model_cfg["rf"]["random_state"],
            n_jobs=model_cfg["rf"]["n_jobs"]
        )

        meta_model = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced"
        )

        return StackingClassifier(
            estimators=[
                ('lr', base_lr),
                ('svm', base_svm),
                ('rf', base_rf)
            ],
            final_estimator=meta_model,
            stack_method="predict_proba",
            n_jobs=-1
        )
    

    raise ValueError(f"Unknown model name: {name}")
