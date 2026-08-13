"""Data audit. Every number in reports/data_audit.md is produced here.

Run: python -m property_price.audit
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data import duplicate_feature_groups, load_raw


@dataclass
class AuditResult:
    n_rows: int
    n_cols: int
    exact_duplicate_rows: int
    duplicate_feature_rows: int
    duplicate_feature_share: float
    n_duplicate_groups: int
    leakage_max_abs_error: float
    leakage_share_within_1: float
    price_skew: float
    log_price_skew: float
    n_locations: int
    other_location_share: float
    built_up_area_equals_area_share: float


def price_per_sqft_identity(df: pd.DataFrame) -> tuple[float, float]:
    """Check price_per_sqft == price * 1e7 / area.

    `price` is in crore (1e7 rupees). If this identity holds, then price_per_sqft
    and area together determine price exactly, and any model given both is reading
    the answer rather than predicting it.
    """
    implied = df["price"] * 1e7 / df["area"]
    err = (implied - df["price_per_sqft"]).abs()
    return float(err.max()), float((err < 1.0).mean())


def run_audit(cfg: Config | None = None) -> AuditResult:
    cfg = cfg or Config.load()
    df = load_raw(cfg)
    max_err, share = price_per_sqft_identity(df)
    groups = duplicate_feature_groups(df, cfg.feature_names)
    dup_feature = int(df.duplicated(subset=cfg.feature_names).sum())

    return AuditResult(
        n_rows=len(df),
        n_cols=df.shape[1],
        exact_duplicate_rows=int(df.duplicated().sum()),
        duplicate_feature_rows=dup_feature,
        duplicate_feature_share=round(dup_feature / len(df), 4),
        n_duplicate_groups=int(groups.nunique()),
        leakage_max_abs_error=round(max_err, 4),
        leakage_share_within_1=round(share, 4),
        price_skew=round(float(df["price"].skew()), 3),
        log_price_skew=round(float(np.log1p(df["price"]).skew()), 3),
        n_locations=int(df["location"].nunique()),
        other_location_share=round(float(df["location"].eq("other").mean()), 4),
        built_up_area_equals_area_share=round(
            float(df["built_up_area"].eq(df["area"]).mean()), 4
        ),
    )


def main() -> None:
    cfg = Config.load()
    result = run_audit(cfg)
    out = cfg.path("reports") / "data_audit.json"
    out.write_text(json.dumps(asdict(result), indent=2))
    print(json.dumps(asdict(result), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
