from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.I)
IP_URL_RE = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}", re.I)
SHORTENER_RE = re.compile(r"(?:bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly)", re.I)
SUSPICIOUS_TERMS = (
    "urgent", "verify", "verification", "account", "password", "login", "suspend",
    "click", "confirm", "immediately", "limited time", "security alert", "invoice",
    "wire transfer", "gift card", "claim", "winner", "prize", "bank", "paypal",
)


def extract_handcrafted_features(texts: pd.Series | list[str]) -> np.ndarray:
    """Extract interpretable URL, structure, and phishing-language signals."""
    rows = []
    for raw in texts:
        text = str(raw)
        lower = text.lower()
        urls = URL_RE.findall(text)
        words = re.findall(r"\b\w+\b", text)
        rows.append([
            len(text),
            len(words),
            len(urls),
            sum(len(u) for u in urls),
            len(IP_URL_RE.findall(text)),
            len(SHORTENER_RE.findall(text)),
            sum(lower.count(term) for term in SUSPICIOUS_TERMS),
            lower.count("<html") + lower.count("<a ") + lower.count("href="),
            text.count("!"),
            text.count("$"),
            sum(ch.isdigit() for ch in text),
            sum(ch.isupper() for ch in text),
            text.count("@"),
            text.count(".") if text else 0,
        ])
    return np.asarray(rows, dtype=np.float64)


def load_dataset(path: Path) -> tuple[pd.Series, np.ndarray]:
    df = pd.read_csv(path, usecols=["Email Text", "Email Type"], nrows=6_000)
    required = {"Email Text", "Email Type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df[["Email Text", "Email Type"]].dropna()
    df["Email Text"] = df["Email Text"].astype(str)
    labels = df["Email Type"].map({"Safe Email": 0, "Phishing Email": 1})
    keep = labels.notna() & df["Email Text"].str.strip().ne("")
    return df.loc[keep, "Email Text"], labels.loc[keep].to_numpy(dtype=np.int8)


def build_features(train_text: pd.Series, other_text: pd.Series, word_vectorizer, char_vectorizer, scaler):
    word_train = word_vectorizer.fit_transform(train_text)
    word_other = word_vectorizer.transform(other_text)
    char_train = char_vectorizer.fit_transform(train_text)
    char_other = char_vectorizer.transform(other_text)
    numeric_train = scaler.fit_transform(extract_handcrafted_features(train_text))
    numeric_other = scaler.transform(extract_handcrafted_features(other_text))
    return (
        hstack([word_train, char_train, csr_matrix(numeric_train)], format="csr"),
        hstack([word_other, char_other, csr_matrix(numeric_other)], format="csr"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a phishing email classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/Phishing_Email.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-samples", type=int, default=4_000, help="Maximum real corpus rows to use; set 0 for all rows.")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)

    texts, y = load_dataset(args.data)
    if args.max_samples and len(texts) > args.max_samples:
        texts, _, y, _ = train_test_split(texts, y, train_size=args.max_samples, random_state=42, stratify=y)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.20, random_state=42, stratify=y
    )

    word_vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        min_df=2, max_df=0.98, max_features=8_000, sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=3,
        max_features=2_000, sublinear_tf=True,
    )
    scaler = StandardScaler()
    x_train_features, x_test_features = build_features(
        x_train, x_test, word_vectorizer, char_vectorizer, scaler
    )

    model = LogisticRegression(max_iter=1_000, class_weight="balanced", solver="liblinear", random_state=42)
    model.fit(x_train_features, y_train)
    predictions = model.predict(x_test_features)

    accuracy = float(accuracy_score(y_test, predictions))
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(y_test, predictions, target_names=["Safe", "Phishing"], output_dict=True)
    print(f"Samples: {len(texts):,} | Train: {len(x_train):,} | Test: {len(x_test):,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=["Safe", "Phishing"]))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Phishing"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Phishing Email Detector — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.artifacts / "confusion_matrix.png", dpi=180)
    plt.close()

    bundle = {
        "model": model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "scaler": scaler,
        "labels": {0: "Safe", 1: "Phishing"},
    }
    joblib.dump(bundle, args.artifacts / "phishing_email_model.joblib", compress=3)
    metrics = {
        "dataset": str(args.data), "random_state": 42, "test_size": 0.2,
        "samples": int(len(texts)), "train_samples": int(len(x_train)), "test_samples": int(len(x_test)),
        "class_counts": {"Safe": int((y == 0).sum()), "Phishing": int((y == 1).sum())},
        "accuracy": accuracy, "confusion_matrix": cm.tolist(), "classification_report": report,
        "feature_blocks": {"word_tfidf": int(x_train_features.shape[1] - char_vectorizer.transform(x_train.head(1)).shape[1] - 14), "char_tfidf": int(char_vectorizer.transform(x_train.head(1)).shape[1]), "handcrafted": 14},
    }
    (args.artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
