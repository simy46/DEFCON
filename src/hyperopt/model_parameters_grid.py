

from config.consts import LINEAR_SVM, LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST


def build_model(model_cfg):
    name = model_cfg["name"]

    if name == LOGISTIC_REGRESSION:
        return {}

    if name == LINEAR_SVM:
        return {}

    if name == RANDOM_FOREST:
        return {
            "n_estimators": [100, 300, 500, 800, 1200],
            "max_depth": [None, 6, 10, 20, 40],
            "min_samples_split": [2, 5, 10, 20, 40],
            "min_samples_leaf": [1, 2, 4, 10, 20],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 1.0],
            "criterion": ["gini", "entropy"],
            "bootstrap": [True, False]
        }

    if name == XGBOOST:
        return {}