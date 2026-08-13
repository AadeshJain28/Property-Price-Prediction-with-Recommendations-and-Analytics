from __future__ import annotations

import numpy as np

from property_price.evaluate import baseline_scores, r2, score


def test_r2_of_perfect_prediction_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == 1.0


def test_r2_of_mean_predictor_is_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(r2(y, np.full_like(y, y.mean()))) < 1e-12


def test_r2_hand_computed():
    # y = [1,2,3], yhat = [1,2,4]. SSres = 1, SStot = 2 -> R2 = 0.5
    assert abs(r2(np.array([1.0, 2, 3]), np.array([1.0, 2, 4])) - 0.5) < 1e-12


def test_log_and_price_spaces_are_reported_separately():
    y_price = np.array([0.5, 1.0, 10.0])
    pred = np.array([0.6, 1.1, 8.0])
    s = score(y_price, pred, np.log1p(y_price), np.log1p(pred))
    assert s.r2_log != s.r2_price, "the two spaces must not be conflated"
    assert s.mape_price > 0


def test_baselines_are_beatable_but_not_trivial():
    rng = np.random.default_rng(0)
    y_tr = rng.lognormal(0, 1, 500)
    y_te = rng.lognormal(0, 1, 200)
    out = baseline_scores(y_tr, y_te)
    assert set(out) == {"baseline_mean", "baseline_median"}
    for s in out.values():
        assert s.r2_price <= 0.05, "a constant predictor must not look good"
