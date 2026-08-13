"""Feature assembly and the leakage guard."""

from __future__ import annotations

import pandas as pd

from .config import Config


class LeakageError(ValueError):
    """Raised when a banned column reaches the model matrix."""


def assert_no_banned_features(columns: list[str], cfg: Config) -> None:
    """Refuse a *feature list* containing a column that determines the target.

    `columns` is the set of columns about to be used as model inputs -- not the
    columns present in the source frame. The raw data legitimately contains
    `price_per_sqft` and `area`; the audit is computed from them. What must never
    happen is their selection as features.

    This is deliberately an exception rather than a silent drop. A silent drop
    would let a future edit re-add `price_per_sqft` to config and never notice
    that the resulting R^2 is meaningless.
    """
    banned = sorted(set(columns) & set(cfg.banned_features))
    if banned:
        raise LeakageError(
            f"{banned} determine(s) the target exactly "
            f"(price_per_sqft = price * 1e7 / area); refusing to build features."
        )


def build_matrix(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Select the configured features, then assert the result is clean.

    The guard runs on the *output* as a post-condition. Running it on `df.columns`
    instead -- as the first version of this function did -- rejected every raw
    frame, because the raw frame is supposed to carry the banned columns. See the
    correction table in reports/data_audit.md.
    """
    missing = sorted(set(cfg.feature_names) - set(df.columns))
    if missing:
        raise ValueError(f"frame is missing configured features: {missing}")

    matrix = df[cfg.feature_names].copy()
    assert_no_banned_features(list(matrix.columns), cfg)
    return matrix


class SchemaError(ValueError):
    """Raised when a frame cannot be coerced to the declared schema."""


def coerce_schema(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Cast a frame to the declared schema.

    ML-8 shipped a train/serve skew bug -- DuckDB typed its Yes/No columns as BOOLEAN,
    so the numeric/categorical split differed between training and serving. This project
    is not exposed to that failure (it trains and serves from the same CSV, and the split
    is declared in config rather than inferred), but the serving path deserves the same
    guarantee rather than relying on that remaining true.

    Numerics go through `pd.to_numeric` *without* forcing float, so an int column stays
    int and the existing fitted artefact remains valid. Categoricals become `str`.
    """
    out = df.copy()

    for col in cfg.categorical_features:
        if col not in out.columns:
            raise SchemaError(f"missing declared categorical column: {col}")
        if out[col].isna().any():
            raise SchemaError(f"{col} contains nulls; the model has no category for them")
        out[col] = out[col].astype(str)

    for col in cfg.numeric_features:
        if col not in out.columns:
            raise SchemaError(f"missing declared numeric column: {col}")
        coerced = pd.to_numeric(out[col], errors="coerce")
        if coerced.isna().any() and not out[col].isna().any():
            bad = out.loc[coerced.isna(), col].unique()[:5]
            raise SchemaError(f"{col} declared numeric but holds non-numeric values: {bad}")
        out[col] = coerced

    return out


def prepare_inference_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The serving-side counterpart of `build_matrix`.

    The dashboard and the API both route through this, so neither can compose its own
    interpretation of the model's inputs.
    """
    return coerce_schema(df, cfg)[cfg.feature_names]


def make_preprocessor(cfg: Config):
    """One-hot for categoricals, passthrough scaling for numerics.

    Tree ensembles do not need the scaler, but the linear baselines in the
    leaderboard do, and sharing one preprocessor keeps the comparison fair.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cfg.numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
                cfg.categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
