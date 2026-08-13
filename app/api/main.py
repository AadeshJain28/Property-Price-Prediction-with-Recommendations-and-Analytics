"""FastAPI scoring service.

    uvicorn app.api.main:app --reload

Endpoints
    GET  /health          liveness + whether the model artefact loaded
    POST /predict         one property -> predicted price (INR crore)
    POST /predict/batch   many properties
    GET  /recommend/{i}   top-k similar properties for a row index
    GET  /metadata        the metrics this model actually achieved
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "price_model.joblib"
RECO_PATH = ROOT / "models" / "recommender.joblib"
SUMMARY_PATH = ROOT / "reports" / "summary.json"

app = FastAPI(
    title="Bengaluru Property Price API",
    version="0.1.0",
    description="Price prediction and content-based recommendation.",
)


class Property(BaseModel):
    property_type: Literal["flat", "house"] = "flat"
    availability: str = Field(default="Ready To Move")
    location: str = Field(..., examples=["whitefield"])
    area_type: str = Field(default="Super built-up  Area")
    bedroom: int = Field(..., ge=1, le=20)
    bath: int = Field(..., ge=1, le=20)
    balcony: int = Field(default=1, ge=0, le=10)
    built_up_area: float = Field(..., gt=100, lt=50_000, description="square feet")


class Prediction(BaseModel):
    price_crore: float
    price_inr: float
    model_version: str


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found -- run `make train` before starting the API."
        )
    import joblib

    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_recommender():
    if not RECO_PATH.exists():
        raise FileNotFoundError(f"{RECO_PATH} not found -- run `make train` first.")
    import joblib

    return joblib.load(RECO_PATH)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_present": MODEL_PATH.exists(),
        "recommender_present": RECO_PATH.exists(),
    }


@app.get("/metadata")
def metadata() -> dict:
    """Serve the measured metrics alongside the model, so a consumer of the API
    can see what it is and is not good at without reading the README."""
    if not SUMMARY_PATH.exists():
        raise HTTPException(404, "no summary.json -- run `make train`")
    return json.loads(SUMMARY_PATH.read_text())


def _predict_frame(rows: list[Property]) -> np.ndarray:
    """Pydantic payload -> the exact schema the pipeline was fitted on.

    Shares `prepare_inference_frame` with the dashboard so the two cannot drift.
    """
    from property_price.config import Config
    from property_price.features import prepare_inference_frame

    bundle = _load_model()
    frame = prepare_inference_frame(
        pd.DataFrame([r.model_dump() for r in rows]), Config.load()
    )
    pred_log = bundle["pipeline"].predict(frame)
    return np.expm1(pred_log)


@app.post("/predict", response_model=Prediction)
def predict(item: Property) -> Prediction:
    try:
        crore = float(_predict_frame([item])[0])
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    return Prediction(price_crore=round(crore, 4), price_inr=round(crore * 1e7, 2), model_version="0.1.0")


@app.post("/predict/batch")
def predict_batch(items: list[Property]) -> dict:
    if not items:
        raise HTTPException(422, "empty batch")
    if len(items) > 1000:
        raise HTTPException(413, "batch limit is 1000 rows")
    try:
        preds = _predict_frame(items)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"n": len(items), "price_crore": [round(float(p), 4) for p in preds]}


@app.get("/recommend/{index}")
def recommend(index: int, k: int = 5) -> dict:
    try:
        rec = _load_recommender()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    if not 0 <= index < len(rec.frame_):
        raise HTTPException(404, f"index out of range 0..{len(rec.frame_) - 1}")
    out = rec.recommend(index, k)
    return {"query": rec.frame_.iloc[index].to_dict(), "recommendations": out.to_dict("records")}
