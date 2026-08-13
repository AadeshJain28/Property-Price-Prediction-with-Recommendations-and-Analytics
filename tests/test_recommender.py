from __future__ import annotations

import numpy as np
import pytest

from property_price.recommend import PropertyRecommender

WEIGHTS = {"location": 0.4, "features": 0.4, "price": 0.2}


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        PropertyRecommender({"location": 0.5, "features": 0.4, "price": 0.2})


def test_never_recommends_the_query_itself(raw):
    rec = PropertyRecommender(WEIGHTS).fit(raw.head(400))
    out = rec.recommend(7, k=5)
    assert 7 not in out.index
    assert len(out) == 5


def test_similarity_is_finite_except_at_the_query(raw):
    rec = PropertyRecommender(WEIGHTS).fit(raw.head(200))
    sims = rec.similarity(3)
    assert np.isneginf(sims[3])
    assert np.isfinite(np.delete(sims, 3)).all()


def test_recommendations_beat_random_on_the_stated_proxy(raw):
    """The evaluation must show the recommender doing something -- a random
    shortlist would score near the base rate of same-location listings."""
    sample = raw.sample(1500, random_state=0).reset_index(drop=True)
    rec = PropertyRecommender(WEIGHTS).fit(sample)
    scores = rec.evaluate(n_queries=120, seed=1)
    base_rate = float((sample.location.value_counts() / len(sample)).pow(2).sum())
    assert scores.location_hit_rate > base_rate * 5
    assert 0.0 <= scores.precision_at_k <= 1.0
