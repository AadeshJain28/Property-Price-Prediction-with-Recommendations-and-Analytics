"""Streamlit dashboard: prediction, analytics and recommendations.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Bengaluru Property Analytics", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load the price model; build the recommender in memory.

    The price model is a 1.7 MB pickle and ships with the repo. The recommender
    pickles the entire frame plus its encoded matrices and comes to 22 MB, which is
    a poor thing to keep in git for an object that refits in about a second. It is
    rebuilt here and cached for the life of the process.
    """
    import joblib

    model_path = ROOT / "models" / "price_model.joblib"
    model = joblib.load(model_path) if model_path.exists() else None

    reco = None
    try:
        from property_price.config import Config
        from property_price.recommend import PropertyRecommender

        cfg = Config.load()
        reco = PropertyRecommender(
            cfg.raw["recommender"]["weights"], cfg.raw["recommender"]["top_k"]
        ).fit(pd.read_csv(ROOT / "data" / "raw" / "final_data.csv"))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Recommender unavailable: {exc}")

    return model, reco


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "raw" / "final_data.csv")


df = load_data()
model, reco = load_artifacts()

st.title("Bengaluru Property Price — prediction, analytics, recommendations")

if model is None:
    st.warning("No trained model found. Run `make train` first; analytics still work below.")

tab_predict, tab_analytics, tab_reco, tab_quality = st.tabs(
    ["Predict", "Analytics", "Recommend", "Model quality"]
)

with tab_predict:
    c1, c2, c3 = st.columns(3)
    with c1:
        location = st.selectbox("Location", sorted(df.location.unique()))
        property_type = st.selectbox("Property type", sorted(df.property_type.unique()))
    with c2:
        bedroom = st.number_input("Bedrooms", 1, 20, 3)
        bath = st.number_input("Bathrooms", 1, 20, 2)
        # `availability` is a model feature. It used to be hardcoded to "Ready To Move",
        # which silently scored every property as ready-to-move regardless of what the
        # user meant. Exposed so the input matches the prediction.
        availability = st.selectbox("Availability", sorted(df.availability.unique()))
    with c3:
        balcony = st.number_input("Balconies", 0, 10, 1)
        built_up_area = st.number_input("Built-up area (sqft)", 200.0, 20000.0, 1200.0, step=50.0)
        area_type = st.selectbox(
            "Area type", sorted(df.area_type.unique()),
            index=sorted(df.area_type.unique()).index(df.area_type.mode()[0]),
        )

    if st.button("Predict price", type="primary") and model is not None:
        from property_price.config import Config
        from property_price.features import prepare_inference_frame

        cfg = Config.load()
        row = prepare_inference_frame(pd.DataFrame([{
            "property_type": property_type,
            "availability": availability,
            "location": location,
            "area_type": area_type,
            "bedroom": bedroom, "bath": bath, "balcony": balcony,
            "built_up_area": built_up_area,
        }]), cfg)
        crore = float(np.expm1(model["pipeline"].predict(row))[0])
        st.metric("Predicted price", f"Rs. {crore:.2f} crore")
        comparable = df[(df.location == location) & (df.bedroom == bedroom)]
        if len(comparable) >= 5:
            lo, hi = comparable.price.quantile([0.25, 0.75])
            st.caption(
                f"{len(comparable)} comparable listings in {location}: "
                f"interquartile range Rs. {lo:.2f}–{hi:.2f} crore."
            )
        else:
            st.caption(
                f"Only {len(comparable)} comparable listings — treat this estimate as weak."
            )

with tab_analytics:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Price distribution")
        st.bar_chart(np.log1p(df.price).round(1).value_counts().sort_index())
        st.caption("Log scale: raw price skew is 6.5, so a linear axis is unreadable.")
    with c2:
        st.subheader("Median price by location (top 20 by volume)")
        top = df.location.value_counts().head(20).index
        st.bar_chart(df[df.location.isin(top)].groupby("location").price.median().sort_values())
    st.subheader("Price per sqft by bedroom count")
    st.dataframe(
        df.assign(pps=df.price * 1e7 / df.area)
        .groupby("bedroom")
        .agg(listings=("price", "size"), median_price_cr=("price", "median"), median_pps=("pps", "median"))
        .round(2)
    )

with tab_reco:
    if reco is None:
        st.info("Recommender artefact missing — run `make train`.")
    else:
        idx = st.number_input("Query listing index", 0, len(df) - 1, 0)
        k = st.slider("How many recommendations", 1, 20, 5)
        st.write("**Query listing**")
        st.dataframe(df.iloc[[idx]])
        st.write("**Recommended**")
        st.dataframe(reco.recommend(int(idx), k))

with tab_quality:
    summary_path = ROOT / "reports" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        st.json(summary)
    else:
        st.info("No reports/summary.json yet — run `make train`.")
    audit_path = ROOT / "reports" / "data_audit.json"
    if audit_path.exists():
        st.subheader("Data audit")
        st.json(json.loads(audit_path.read_text()))
