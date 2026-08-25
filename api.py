from __future__ import annotations

import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.sparse import csr_matrix, hstack

from train_model import extract_handcrafted_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/phishing_email_model.joblib"))
app = FastAPI(title="Phishing Email Detection API", version="1.0.0")
_bundle = None


class EmailRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200_000, description="Raw email subject/body text")


class ScanResponse(BaseModel):
    label: str
    confidence: float
    phishing_probability: float
    model: str


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def scan_text(text: str) -> ScanResponse:
    bundle = get_bundle()
    word = bundle["word_vectorizer"].transform([text])
    char = bundle["char_vectorizer"].transform([text])
    numeric = bundle["scaler"].transform(extract_handcrafted_features([text]))
    features = hstack([word, char, csr_matrix(numeric)], format="csr")
    model = bundle["model"]
    label_id = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    return ScanResponse(
        label=bundle["labels"][label_id],
        confidence=float(probabilities[label_id]),
        phishing_probability=float(probabilities[1]),
        model="tfidf-word-char+url-heuristics/logistic-regression",
    )


@app.get("/health")
def health():
    try:
        get_bundle()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/scan", response_model=ScanResponse)
def scan_email(request: EmailRequest):
    try:
        return scan_text(request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc
