"""FastAPI service for the Group 3 early dropout-risk model.

Run from the project root:   uvicorn api.main:app --port 8000
Docs (auto-generated):       http://localhost:8000/docs

The model, threshold, validation and explanations all come from predictor.py,
so the API and the Streamlit app score students identically.
Decision support only: scores must not be used alone for consequential decisions.
"""
import hashlib
import os
import time
from functools import lru_cache
from typing import Optional

import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from predictor import ArtifactError, DropoutPredictor, InputError

app = FastAPI(
    title="Student Dropout Risk API",
    description="Estimates end-of-semester-1 dropout risk for decision support. Not for automated decisions.",
    version="1.0.0",
)

MAX_BATCH = 1000


@lru_cache(maxsize=1)
def get_predictor() -> DropoutPredictor:
    """Load the model once, on first use (not per request)."""
    predictor = DropoutPredictor()
    if predictor.is_synthetic and os.environ.get("ALLOW_SYNTHETIC") != "1":
        raise ArtifactError("Refusing to serve a SYNTHETIC demo model (set ALLOW_SYNTHETIC=1 for tests only).")
    return predictor


@lru_cache(maxsize=1)
def model_sha256() -> str:
    p = get_predictor()
    path = p.artifact_dir / "dropout_pipeline_v1.joblib"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- schemas
class Student(BaseModel):
    student_id: Optional[str] = Field(None, description="Optional identifier echoed back in the response")
    features: dict[str, float] = Field(
        ..., description="The 30 semester-1 fields, keyed by their exact column names (see /schema)"
    )


class BatchRequest(BaseModel):
    students: list[Student] = Field(..., min_length=1, max_length=MAX_BATCH)


class Factor(BaseModel):
    feature: str
    value: float
    contribution: float


class Prediction(BaseModel):
    student_id: Optional[str]
    risk_score: float
    flagged: bool
    predicted_class: str
    main_risk_factors: list[Factor]


class PredictResponse(Prediction):
    threshold: float
    model_version: str
    notes: list[str]


class BatchResponse(BaseModel):
    count: int
    flagged: int
    threshold: float
    model_version: str
    notes: list[str]
    predictions: list[Prediction]


# ---------------------------------------------------------------- helpers
def _load():
    try:
        return get_predictor()
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")


def _score(predictor: DropoutPredictor, students: list[Student]):
    rows = [s.features for s in students]
    df = pd.DataFrame(rows)
    try:
        scored = predictor.predict(df)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:  # never leak internals
        raise HTTPException(status_code=500, detail="Internal error while scoring.")

    notes = list(scored.notes)
    contrib = predictor.contributions(scored.features) if predictor.can_explain else None
    if contrib is not None and not predictor.explanations_centered:
        contrib = None
        notes.append("Risk-factor explanations omitted: the model metadata has no feature_means, "
                     "so contributions would not be relative to the average student.")

    preds = []
    for i, s in enumerate(students):
        r = scored.results.iloc[i]
        factors = []
        if contrib is not None:
            up, _ = predictor.top_factors(contrib.iloc[i], scored.features.iloc[i], k=3)
            factors = [Factor(feature=f.feature, value=float(f.value), contribution=round(float(f.contribution), 3))
                       for f in up.itertuples()]
        preds.append(Prediction(student_id=s.student_id, risk_score=float(r["risk_score"]),
                                flagged=bool(r["flagged"]), predicted_class=str(r["predicted_class"]),
                                main_risk_factors=factors))
    return preds, notes


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    try:
        p = get_predictor()
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")
    return {"status": "healthy", "model": "dropout_pipeline_v1", "threshold": p.threshold,
            "synthetic": p.is_synthetic, "model_sha256": model_sha256(),
            "sklearn_runtime": sklearn.__version__, "sklearn_trained": p.meta.get("sklearn_version")}


@app.get("/schema")
def schema():
    """The exact field names the model expects, with the valid values it learned."""
    p = _load()
    return {"features": p.features, "binary_0_1": p.binary,
            "categorical_codes_seen_in_training": p.categories, "numeric": p.numeric}


@app.post("/predict", response_model=PredictResponse)
def predict(student: Student):
    p = _load()
    preds, notes = _score(p, [student])
    return PredictResponse(**preds[0].model_dump(), threshold=p.threshold, model_version="v1", notes=notes)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    p = _load()
    start = time.time()
    preds, notes = _score(p, request.students)
    return BatchResponse(count=len(preds), flagged=sum(x.flagged for x in preds),
                         threshold=p.threshold, model_version="v1", notes=notes, predictions=preds)
