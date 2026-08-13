"""Loading and the grouped split.

The split is the load-bearing part of this module. See `duplicate_feature_groups`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def load_raw(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path("raw"))
    missing = set(cfg.feature_names + [cfg.target]) - set(df.columns)
    if missing:
        raise ValueError(f"raw data is missing required columns: {sorted(missing)}")
    return df


def duplicate_feature_groups(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Assign every row a group id; rows with an identical feature vector share one.

    14.3% of rows in this dataset repeat a feature vector already present under a
    different price -- the same flat listed twice, or two indistinguishable flats.
    Under a shuffled split those near-copies land on both sides and the model is
    scored partly on rows it memorised. Grouping keeps each cluster in one fold.
    """
    key = df[feature_cols].astype(str).agg("|".join, axis=1)
    return key.map({k: i for i, k in enumerate(key.unique())})


def grouped_train_test_split(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by duplicate-feature group, so no group straddles the boundary."""
    from sklearn.model_selection import GroupShuffleSplit

    groups = duplicate_feature_groups(df, cfg.feature_names)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.raw["split"]["test_size"],
        random_state=cfg.raw["split"]["seed"],
    )
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def naive_train_test_split(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The shuffled split the original notebook used, kept for the leakage comparison.

    Reported side by side with the grouped split in reports/split_comparison.md so the
    size of the optimism is a measured number rather than an assertion.
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        df,
        test_size=cfg.raw["split"]["test_size"],
        random_state=cfg.raw["split"]["seed"],
    )


def target_vectors(df: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_price, y_model) where y_model is the transformed target."""
    y = df[cfg.target].to_numpy(dtype=float)
    if cfg.raw["target_transform"] == "log1p":
        return y, np.log1p(y)
    return y, y


def inverse_target(y_model: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.raw["target_transform"] == "log1p":
        return np.expm1(y_model)
    return y_model
