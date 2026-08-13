"""Model zoo and the tuned estimator."""

from __future__ import annotations

from typing import Any


def model_zoo() -> dict[str, Any]:
    """The regressors compared on the leaderboard.

    Kept deliberately wide: the linear models are not competitive but they set the
    floor that makes the ensemble numbers interpretable.
    """
    from sklearn.ensemble import (
        AdaBoostRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=None, alpha=1.0),
        "lasso": Lasso(alpha=0.001, max_iter=10_000),
        "svr": SVR(),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "adaboost": AdaBoostRegressor(random_state=42),
        "xgboost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def search_space() -> dict[str, list]:
    return {
        "regressor__n_estimators": [200, 300, 400, 600],
        "regressor__max_depth": [4, 5, 6, 7, 8],
        "regressor__learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2],
        "regressor__subsample": [0.6, 0.8, 1.0],
        "regressor__colsample_bytree": [0.6, 0.8, 1.0],
        "regressor__min_child_weight": [1, 3, 5],
        "regressor__gamma": [0, 0.1, 0.3],
    }


def build_pipeline(cfg, estimator):
    from sklearn.pipeline import Pipeline

    from .features import make_preprocessor

    return Pipeline(
        [("preprocessor", make_preprocessor(cfg)), ("regressor", estimator)]
    )
