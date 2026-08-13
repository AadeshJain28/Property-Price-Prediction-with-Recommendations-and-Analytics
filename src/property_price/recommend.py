"""Content-based property recommender with an evaluation protocol.

The original notebook built a cosine-similarity recommender and stopped there --
no metric, so "it recommends similar properties" was untestable. Similarity
recommenders have no ground-truth labels here, so this module defines a proxy
relevance rule up front and scores against it. The rule is stated in the code so
a reader can disagree with it, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RecommenderScores:
    k: int
    precision_at_k: float
    location_hit_rate: float
    median_price_gap_pct: float
    n_queries: int


class PropertyRecommender:
    """Blended cosine similarity over location, structural features and price."""

    def __init__(self, weights: dict[str, float], top_k: int = 5) -> None:
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1.0")
        self.weights = weights
        self.top_k = top_k
        self.frame_: pd.DataFrame | None = None

    # -- fitting ---------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> PropertyRecommender:
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        self.frame_ = df.reset_index(drop=True)
        self._loc_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._loc = self._loc_enc.fit_transform(self.frame_[["location"]])

        self._feat_scaler = StandardScaler()
        self._feat = self._feat_scaler.fit_transform(
            self.frame_[["bedroom", "bath", "balcony", "built_up_area"]]
        )

        self._price_scaler = StandardScaler()
        self._price = self._price_scaler.fit_transform(
            np.log1p(self.frame_[["price"]].to_numpy(dtype=float))
        )
        return self

    # -- querying --------------------------------------------------------------
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        num = b @ a
        den = np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-12
        return num / den

    def similarity(self, idx: int) -> np.ndarray:
        sims = (
            self.weights["location"] * self._cosine(self._loc[idx], self._loc)
            + self.weights["features"] * self._cosine(self._feat[idx], self._feat)
            + self.weights["price"] * self._cosine(self._price[idx], self._price)
        )
        sims[idx] = -np.inf  # never recommend the query back to itself
        return sims

    def recommend(self, idx: int, k: int | None = None) -> pd.DataFrame:
        k = k or self.top_k
        sims = self.similarity(idx)
        top = np.argsort(sims)[::-1][:k]
        out = self.frame_.iloc[top].copy()
        out["similarity"] = sims[top]
        return out

    # -- evaluation ------------------------------------------------------------
    def evaluate(self, n_queries: int = 300, k: int | None = None, seed: int = 42):
        """Score against a stated proxy for relevance.

        A recommendation counts as relevant when it shares the query's location
        AND sits within 25% of its price -- i.e. it is a property the same buyer
        could plausibly consider. This is a proxy, not ground truth; it is written
        down so the number means something specific.
        """
        k = k or self.top_k
        rng = np.random.default_rng(seed)
        n = len(self.frame_)
        queries = rng.choice(n, size=min(n_queries, n), replace=False)

        hits, loc_hits, gaps = [], [], []
        for q in queries:
            rec = self.recommend(int(q), k)
            qloc = self.frame_.location.iloc[q]
            qprice = float(self.frame_.price.iloc[q])
            same_loc = rec.location.eq(qloc).to_numpy()
            gap = (rec.price.to_numpy(dtype=float) - qprice) / qprice
            hits.append(float((same_loc & (np.abs(gap) <= 0.25)).mean()))
            loc_hits.append(float(same_loc.mean()))
            gaps.extend(np.abs(gap).tolist())

        return RecommenderScores(
            k=k,
            precision_at_k=round(float(np.mean(hits)), 4),
            location_hit_rate=round(float(np.mean(loc_hits)), 4),
            median_price_gap_pct=round(float(np.median(gaps) * 100), 2),
            n_queries=len(queries),
        )
