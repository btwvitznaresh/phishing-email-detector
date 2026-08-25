from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from scipy.sparse import csr_matrix, hstack

from train_model import extract_handcrafted_features


def predict_email(text: str, artifact_path: Path = Path("artifacts/phishing_email_model.joblib")) -> tuple[str, float]:
    bundle = joblib.load(artifact_path)
    word = bundle["word_vectorizer"].transform([text])
    char = bundle["char_vectorizer"].transform([text])
    numeric = bundle["scaler"].transform(extract_handcrafted_features([text]))
    features = hstack([word, char, csr_matrix(numeric)], format="csr")
    label_id = int(bundle["model"].predict(features)[0])
    probability = float(bundle["model"].predict_proba(features)[0, label_id])
    return bundle["labels"][label_id], probability


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an email as Safe or Phishing.")
    parser.add_argument("email", nargs="?", help="Email text. If omitted, read from stdin.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/phishing_email_model.joblib"))
    args = parser.parse_args()
    text = args.email if args.email is not None else input("Paste email text: ")
    label, confidence = predict_email(text, args.model)
    print(f"Prediction: {label} (model confidence: {confidence:.2%})")
