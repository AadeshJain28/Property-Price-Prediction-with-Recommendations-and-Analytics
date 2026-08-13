from __future__ import annotations

import pandas as pd

from property_price.data import duplicate_feature_groups, grouped_train_test_split


def test_duplicate_groups_on_hand_built_frame(toy, cfg):
    groups = duplicate_feature_groups(toy, cfg.feature_names)
    # rows 0 and 1 are identical on features -> same group; all others distinct
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.nunique() == 5, "6 rows, one duplicated pair -> 5 groups"


def test_grouped_split_never_straddles_a_group(raw, cfg):
    train, test = grouped_train_test_split(raw, cfg)
    g = duplicate_feature_groups(raw, cfg.feature_names)
    train_groups = set(g.iloc[train.index])
    test_groups = set(g.iloc[test.index])
    assert not (train_groups & test_groups), "a duplicate-feature group spans both folds"


def test_split_sizes_are_sane(raw, cfg):
    train, test = grouped_train_test_split(raw, cfg)
    assert len(train) + len(test) == len(raw)
    assert 0.15 < len(test) / len(raw) < 0.25


def test_real_data_has_the_duplicates_the_split_exists_for(raw, cfg):
    """If this drops to zero the grouped split is pointless -- and the README
    claim about 14% duplicates would be stale. Fail loudly rather than drift."""
    dups = raw.duplicated(subset=cfg.feature_names).sum()
    assert dups > 1000, f"expected ~1520 duplicate-feature rows, found {dups}"
