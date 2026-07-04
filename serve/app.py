"""FastAPI inference gate for image safety classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from pipeline.embed import embed_image
from pipeline.schemas import ClassifierRequest, ClassifierResponse
from pipeline.train import EMBED_COLS, load_classifier

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "classifier.pkl"

app = FastAPI(title="Image Safety Classifier", version="0.1.0")
_learner = None
_version = "untrained"
_threshold = 0.5


@app.on_event("startup")
def load_model() -> None:
    global _learner, _version, _threshold
    if ARTIFACT.exists():
        _learner, _version, _threshold = load_classifier(ARTIFACT)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _learner is not None, "version": _version}


@app.post("/classify", response_model=ClassifierResponse)
def classify(req: ClassifierRequest) -> ClassifierResponse:
    if _learner is None:
        raise HTTPException(503, "Model not trained. Run `run-pipeline` first.")
    path = Path(req.image_path)
    if not path.exists():
        raise HTTPException(400, f"Image not found: {path}")

    embedding = embed_image(path)
    row = pd.DataFrame([embedding], columns=EMBED_COLS)
    test_dl = _learner.dls.test_dl(row)
    preds, _ = _learner.get_preds(dl=test_dl)
    flagged_probability = float(preds[0, 1])
    is_flagged = flagged_probability >= _threshold

    return ClassifierResponse(
        image_path=str(path),
        is_flagged=is_flagged,
        flagged_probability=flagged_probability,
        decision_threshold=_threshold,
        model_version=_version,
    )
