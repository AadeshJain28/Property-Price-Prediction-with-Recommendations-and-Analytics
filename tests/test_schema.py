"""Train/serve schema contract.

The sibling repo (ML-8) shipped a train/serve skew bug: the numeric/categorical split was
inferred with `select_dtypes`, its training reader typed 13 Yes/No columns as BOOLEAN, and
the dashboard fed the same columns in as strings. `StandardScaler` raised
`could not convert string to float: 'Yes'` on the first run of the app.

This project does not have that exposure — it trains and serves from the same CSV, and the
split is declared in `config.yaml` rather than inferred. These tests exist so that stays
true rather than being true by accident.
"""

from __future__ import annotations

import pandas as pd
import pytest

from property_price.features import (
    SchemaError,
    build_matrix,
    coerce_schema,
    prepare_inference_frame,
)


def app_style_row(df: pd.DataFrame) -> pd.DataFrame:
    """A single row composed the way the dashboard composes one."""
    return pd.DataFrame([{
        "property_type": "flat",
        "availability": "Ready To Move",
        "location": "whitefield",
        "area_type": df.area_type.mode()[0],
        "bedroom": 3, "bath": 2, "balcony": 1,
        "built_up_area": 1200.0,
    }])


def test_training_and_serving_frames_agree_on_dtypes(raw, cfg):
    """The check that would have caught ML-8's bug, run from the serving side."""
    train_matrix = build_matrix(raw, cfg)
    serve_matrix = prepare_inference_frame(app_style_row(raw), cfg)

    assert list(train_matrix.columns) == list(serve_matrix.columns)
    for col in cfg.feature_names:
        assert train_matrix[col].dtype.kind == serve_matrix[col].dtype.kind, (
            f"{col}: training is {train_matrix[col].dtype}, "
            f"serving is {serve_matrix[col].dtype}"
        )


def test_serving_frame_has_exactly_the_model_features_in_order(raw, cfg):
    frame = prepare_inference_frame(app_style_row(raw), cfg)
    assert list(frame.columns) == cfg.feature_names
    assert not set(frame.columns) & set(cfg.banned_features)


def test_every_categorical_the_app_can_emit_was_seen_in_training(raw, cfg):
    """The dashboard populates its dropdowns from the training data, so every value it
    can emit must be in the model's vocabulary. If a control is ever hardcoded to a
    literal instead, this catches a typo before a user does."""
    frame = prepare_inference_frame(app_style_row(raw), cfg)
    for col in cfg.categorical_features:
        vocabulary = set(raw[col].astype(str).unique())
        emitted = frame[col].iloc[0]
        assert emitted in vocabulary, f"{col}={emitted!r} is not a level the model saw"


def test_availability_is_a_real_feature_not_a_constant(raw, cfg):
    """Regression guard for a bug in the dashboard.

    `availability` was hardcoded to "Ready To Move" while being a live model feature, so
    every prediction silently assumed ready-to-move. It has more than one level and it
    moves the prediction, so it must be a user control.
    """
    assert "availability" in cfg.feature_names
    assert raw.availability.nunique() > 1, (
        "availability has collapsed to one level; if that is real, drop it from the "
        "feature list rather than leaving a dead input in the UI"
    )


def test_coercion_is_idempotent(raw, cfg):
    once = coerce_schema(raw, cfg)
    pd.testing.assert_frame_equal(once, coerce_schema(once, cfg))


def test_numeric_columns_keep_their_integer_types(raw, cfg):
    """`to_numeric` must not silently promote int to float.

    The fitted artefact was trained on int64 bedroom/bath/balcony; changing that here
    would not break StandardScaler but would make the training and serving frames
    genuinely different objects, which is the thing this module exists to prevent.
    """
    coerced = coerce_schema(raw, cfg)
    for col in cfg.numeric_features:
        assert coerced[col].dtype == raw[col].dtype, (
            f"{col} changed from {raw[col].dtype} to {coerced[col].dtype}"
        )


def test_string_numerics_are_recovered(raw, cfg):
    """A number arriving as text (a JSON payload, a CSV re-read) must still work."""
    mangled = raw.copy()
    mangled["built_up_area"] = mangled["built_up_area"].astype(str)
    coerced = coerce_schema(mangled, cfg)
    assert coerced["built_up_area"].dtype.kind == "f"


def test_missing_column_is_reported(raw, cfg):
    with pytest.raises(SchemaError, match="location"):
        coerce_schema(raw.drop(columns=["location"]), cfg)


def test_non_numeric_value_in_numeric_column_is_rejected(raw, cfg):
    broken = raw.copy()
    broken["built_up_area"] = broken["built_up_area"].astype(object)
    broken.loc[broken.index[0], "built_up_area"] = "twelve hundred"
    with pytest.raises(SchemaError, match="built_up_area"):
        coerce_schema(broken, cfg)


def test_null_categorical_is_rejected_rather_than_stringified(raw, cfg):
    """`astype(str)` turns NaN into the string 'nan', a category the model never saw."""
    broken = raw.copy()
    broken.loc[broken.index[0], "location"] = None
    with pytest.raises(SchemaError, match="location"):
        coerce_schema(broken, cfg)
