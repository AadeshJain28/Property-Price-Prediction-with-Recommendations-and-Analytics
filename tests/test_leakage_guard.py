"""The guard that must never silently stop guarding.

price_per_sqft = price * 1e7 / area, exactly. A model handed price_per_sqft and
area does not predict price, it recomputes it. These tests assert the *mechanism*
(an exception is raised, from the specific check) rather than just that some
model scored badly.
"""

from __future__ import annotations

import numpy as np
import pytest

from property_price.audit import price_per_sqft_identity
from property_price.features import LeakageError, assert_no_banned_features, build_matrix


def test_identity_holds_on_real_data(raw):
    """If this ever fails, the banned-feature list needs revisiting -- the reason
    those columns are banned would no longer be true."""
    max_err, share_within_1 = price_per_sqft_identity(raw)
    assert max_err <= 1.0, "price_per_sqft is no longer an exact function of price/area"
    assert share_within_1 == 1.0


def test_identity_on_hand_computed_row():
    """Hand derivation: 0.46 crore = 4,600,000 INR over 1,140 sqft = 4,035.09 /sqft.
    The dataset rounds to 4035, so the residual must be under 1.0 but not zero."""
    import pandas as pd

    df = pd.DataFrame({"price": [0.46], "area": [1140.0], "price_per_sqft": [4035.0]})
    max_err, share = price_per_sqft_identity(df)
    assert 0.0 < max_err < 1.0
    assert share == 1.0


def test_assert_rejects_a_banned_feature_list(cfg):
    """The helper guards a *feature list*, not a source frame."""
    with pytest.raises(LeakageError, match="price_per_sqft"):
        assert_no_banned_features([*cfg.feature_names, "price_per_sqft"], cfg)


def test_build_matrix_accepts_a_raw_frame_and_drops_the_banned_columns(toy, cfg):
    """Regression test for a real bug.

    The first version of `build_matrix` asserted on `df.columns`, so it raised on
    every raw frame -- the raw data is *supposed* to carry price_per_sqft and area,
    because the audit is computed from them. The guard belongs on the selected
    features, as a post-condition.
    """
    assert set(cfg.banned_features).issubset(toy.columns), "fixture must carry the banned cols"

    matrix = build_matrix(toy, cfg)

    assert list(matrix.columns) == cfg.feature_names
    assert not set(matrix.columns) & set(cfg.banned_features)
    assert len(matrix) == len(toy)


def test_build_matrix_still_raises_when_config_smuggles_a_banned_feature(toy, cfg):
    """The post-condition must remain load-bearing.

    Config.validate normally blocks this, so the guard is only reachable via a
    Config built without validation -- exactly what a careless refactor would do.
    """
    from property_price.config import Config

    smuggled = dict(cfg.raw)
    smuggled["numeric_features"] = [*cfg.numeric_features, "price_per_sqft"]
    bad_cfg = Config(raw=smuggled)  # deliberately not validated

    with pytest.raises(LeakageError, match="price_per_sqft"):
        build_matrix(toy, bad_cfg)


def test_build_matrix_reports_missing_features(toy, cfg):
    with pytest.raises(ValueError, match="missing configured features"):
        build_matrix(toy.drop(columns=["bath"]), cfg)


def test_config_rejects_banned_feature_in_feature_list(cfg):
    """Guard on the guard: if someone adds price_per_sqft to numeric_features,
    Config.validate must refuse to load rather than quietly train on it."""
    from property_price.config import Config

    broken = dict(cfg.raw)
    broken["numeric_features"] = cfg.numeric_features + ["price_per_sqft"]
    with pytest.raises(ValueError, match="banned"):
        Config(raw=broken).validate()


def test_leaked_model_is_near_perfect_so_the_guard_is_load_bearing(raw):
    """Demonstrates what the guard prevents, using closed-form arithmetic only.

    Reconstructing price from the two banned columns recovers the target to rounding
    error -- R^2 indistinguishable from 1. That is the score a "model" would report
    if the guard were removed, which is why the guard exists.
    """
    reconstructed = raw["price_per_sqft"] * raw["area"] / 1e7
    y = raw["price"].to_numpy(float)
    r2 = 1 - ((y - reconstructed) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 > 0.9999, f"expected near-perfect reconstruction, got {r2}"


def test_reconstruction_error_matches_the_rounding_bound_exactly(raw):
    """The residual is rounding, and the bound is derivable rather than guessed.

    `price_per_sqft` is stored rounded to whole rupees, so

        price_per_sqft = round(price * 1e7 / area)   =>   |round(x) - x| <= 0.5

    and the reconstruction error in crore is that residual scaled by `area`:

        |price - price_per_sqft * area / 1e7|  <=  0.5 * area / 1e7

    The bound therefore GROWS with area. An earlier version of this test asserted a
    flat `< 0.01 crore` and failed in CI at 0.0386 -- on a 1,306,800 sqft plot listed
    at 2 rupees/sqft, where the bound is 0.065. The constant was the bug; the data
    was fine. Asserting the derived inequality row by row is both stricter and
    correct, and it cannot go stale if the dataset changes.
    """
    residual = np.abs(raw["price"] - raw["price_per_sqft"] * raw["area"] / 1e7)
    bound = 0.5 * raw["area"] / 1e7

    violations = int((residual > bound + 1e-12).sum())
    assert violations == 0, f"{violations} rows exceed the rounding bound"

    # Guard the guard: the bound must actually be tight somewhere, otherwise this
    # test would keep passing even if the identity stopped holding.
    assert (residual > 0.5 * bound).any(), "bound is loose everywhere; identity may have changed"

    # And the error is negligible where it matters -- typical listings, not land plots.
    typical = raw["area"] < 10_000
    assert (residual[typical] / raw["price"][typical]).max() < 0.01
