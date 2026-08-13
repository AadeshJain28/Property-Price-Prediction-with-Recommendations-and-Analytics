"""Metrics, reported in both the modelling space and the space the user cares about."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Scores:
    r2_log: float
    r2_price: float
    mae_price: float
    rmse_price: float
    mape_price: float
    median_ape_price: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def score(
    y_price: np.ndarray,
    y_price_hat: np.ndarray,
    y_log: np.ndarray,
    y_log_hat: np.ndarray,
) -> Scores:
    """Score in both spaces.

    R^2 computed on log1p(price) is not R^2 on price. The target skew here is 6.5,
    so the log-space figure flatters the model; a headline that omits which space
    it was measured in is not a defensible claim. Both are reported, always.
    """
    y_price = np.asarray(y_price, float)
    y_price_hat = np.asarray(y_price_hat, float)
    ape = np.abs((y_price - y_price_hat) / y_price)
    return Scores(
        r2_log=round(r2(y_log, y_log_hat), 4),
        r2_price=round(r2(y_price, y_price_hat), 4),
        mae_price=round(float(np.abs(y_price - y_price_hat).mean()), 4),
        rmse_price=round(float(np.sqrt(((y_price - y_price_hat) ** 2).mean())), 4),
        mape_price=round(float(ape.mean() * 100), 2),
        median_ape_price=round(float(np.median(ape) * 100), 2),
    )


def baseline_scores(y_train_price: np.ndarray, y_test_price: np.ndarray) -> dict[str, Scores]:
    """Reference points every model must beat to have earned its complexity.

    Without these a leaderboard is unanchored: 0.82 means nothing until you know
    what predicting the training median scores on the same split.
    """
    out: dict[str, Scores] = {}
    for name, const in {
        "mean": float(np.mean(y_train_price)),
        "median": float(np.median(y_train_price)),
    }.items():
        pred = np.full(len(y_test_price), const)
        out[f"baseline_{name}"] = score(
            y_test_price, pred, np.log1p(y_test_price), np.log1p(pred)
        )
    return out
