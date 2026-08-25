# Phishing Email Detection Model

This project trains a binary **Scikit-learn** classifier that labels email text as **Safe** or **Phishing**. It uses a real public corpus with 18,708 labeled rows from the Hugging Face mirror of the Phishing Email Dataset [1]. The implementation combines two TF-IDF text representations with interpretable URL and message-structure features.

## Features

The model uses word n-grams and character n-grams to capture lexical patterns, misspellings, and obfuscated text. It also extracts URL count, URL length, IP-address URLs, URL shorteners, suspicious-language terms, HTML/link markers, exclamation marks, dollar signs, digit count, uppercase count, `@` symbols, and periods. These signals are combined with a balanced logistic-regression classifier.

## Installation

```bash
cd /home/ubuntu/phishing_email_detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
python3 train_model.py
```

The command performs an 80/20 stratified train/test split with a fixed random seed, prints accuracy and a full classification report, writes `artifacts/metrics.json`, saves the fitted model to `artifacts/phishing_email_model.joblib`, and creates `artifacts/confusion_matrix.png`.

## Classify a new email

```bash
python3 predict_email.py "URGENT: Verify your account immediately at https://bit.ly/example"
```

The prediction script also accepts email text from standard input when the positional argument is omitted.

## Important limitation

The reported test score is a benchmark on this corpus, not a guarantee against future campaigns. A production detector should use temporally separated and organization-specific validation data, preserve the original email headers when available, monitor drift, and route uncertain messages for human review. A **Safe** prediction must not be treated as proof that an email is trustworthy.

## Dataset

The default CSV is `data/Phishing_Email.csv`. Its required columns are `Email Text` and `Email Type`, with labels `Safe Email` and `Phishing Email`. Replace this file with another compatible corpus to retrain the system.

## Reference

[1]: https://huggingface.co/datasets/zefang-liu/phishing-email-dataset "zefang-liu/phishing-email-dataset — Hugging Face"

## Real-time FastAPI service

Install the additional API dependencies from `requirements.txt`, then start the service from the project root:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The service exposes `GET /health` and `POST /scan`. FastAPI validates the JSON request body and exposes interactive documentation at `/docs`.

```bash
curl -X POST http://localhost:8000/scan \
  -H 'Content-Type: application/json' \
  -d '{"text":"URGENT: verify your account at https://bit.ly/example"}'
```

The response includes `label`, `confidence`, and `phishing_probability`. For production, place the API behind TLS and authentication, add rate limiting and request logging that excludes raw message bodies, and run it with a production ASGI process manager. The bundled joblib artifact must be treated as trusted code and never replaced with an unverified upload [2].

## Adversarial testing

Run the diagnostic suite with:

```bash
python3 adversarial_tests.py
```

The suite includes Unicode homographs, zero-width characters, defanged URLs, IP-address URLs, URL shorteners, HTML links, and benign controls. Results are written to `artifacts/adversarial_results.json`. This is a hand-curated robustness probe rather than a statistically representative security benchmark. Homograph and obfuscation detection should be strengthened in production with Unicode-script/confusable analysis, URL canonicalization, DNS/domain reputation, redirect resolution in a sandbox, and header/authentication signals.

[2]: https://scikit-learn.org/stable/model_persistence.html "Scikit-learn model persistence and security limitations"
