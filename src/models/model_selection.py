from sklearn.calibration import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier , ExtraTreesClassifier
from xgboost import XGBClassifier # type: ignore
from config.consts import LOGISTIC_REGRESSION, SVM, RANDOM_FOREST, XGBOOST, STACK, LINEAR_SVM, RANDOMIZED_TREE

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
    
    if name == RANDOMIZED_TREE:
        return ExtraTreesClassifier(
            n_estimators=model_cfg["n_estimators"],
            min_samples_split=model_cfg["min_samples_split"],
            min_samples_leaf=model_cfg["min_samples_leaf"],
            max_features=model_cfg["max_features"],
            max_samples=model_cfg["max_samples"],
            bootstrap=True,
            max_depth=model_cfg["max_depth"],
            criterion=model_cfg["criterion"],
            class_weight=model_cfg["class_weight"],
            random_state=model_cfg["random_state"],
            n_jobs=model_cfg["n_jobs"]
        )



    if name == SVM:
        return SVC(
            C=model_cfg["C"],
            kernel='rbf',
            gamma=model_cfg["gamma"],
            shrinking=True,
            probability=model_cfg["probability"],
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

    if name == LINEAR_SVM:
        return LinearSVC(
        penalty=model_cfg["penalty"],
        loss=model_cfg["loss"],
        dual=model_cfg["dual"],
        tol=model_cfg["tol"],
        C=model_cfg["C"],
        fit_intercept=model_cfg["fit_intercept"],
        intercept_scaling=model_cfg["intercept_scaling"],
        class_weight=model_cfg["class_weight"],
        random_state=model_cfg["random_state"],
        max_iter=model_cfg["max_iter"]
        )

    if name == STACK:
        base_lr = build_model(model_cfg["lr"])
        base_svm = build_model(model_cfg["svm"])
        base_rf = build_model(model_cfg["random_forest"])

        base_xgb = XGBClassifier(
            booster=model_cfg["xgboost"]["booster"],
            eta=model_cfg["xgboost"]["eta"],
            max_depth=model_cfg["xgboost"]["max_depth"],
            min_child_weight=model_cfg["xgboost"]["min_child_weight"],
            subsample=model_cfg["xgboost"]["subsample"],
            colsample_bytree=model_cfg["xgboost"]["colsample_bytree"],
            reg_lambda=model_cfg["xgboost"]["reg_lambda"],
            reg_alpha=model_cfg["xgboost"]["reg_alpha"],
            n_estimators=model_cfg["xgboost"]["n_estimators"],
            tree_method=model_cfg["xgboost"]["tree_method"],
            max_bin=model_cfg["xgboost"]["max_bin"],
            n_jobs=model_cfg["xgboost"]["n_jobs"],
            random_state=model_cfg["xgboost"]["random_state"],
            eval_metric=model_cfg["xgboost"]["eval_metric"],
        )

        meta_model = LogisticRegression(
            solver="saga",
            max_iter=5000,
            class_weight="balanced"
        )

        return StackingClassifier(
            estimators=[
                ('lr', base_lr),
                ('svm', base_svm),
                ('rf', base_rf),
                ('xgb', base_xgb)
            ],
            final_estimator=meta_model,
            stack_method="predict_proba",
            passthrough=True,
            n_jobs=-1
        )

    

    raise ValueError(f"Unknown model name: {name}")
