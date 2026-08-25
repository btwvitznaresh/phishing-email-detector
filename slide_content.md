# Phishing Email Detection Model
## Design, performance, and real-time deployment
- Scikit-learn classifier with FastAPI serving
- Public labeled corpus; reproducible training and evaluation

# Problem and objective
- Detect phishing versus safe email from message text and embedded URL signals
- Return a low-latency label and calibrated model probability for downstream review workflows
- Treat the classifier as decision support, not proof of trustworthiness

# Data and modeling pipeline
- Public corpus: 18,708 labeled rows; bounded 4,000-row stratified benchmark used in this sandbox run
- 80/20 stratified split, random state 42
- Word TF-IDF + character TF-IDF + 14 handcrafted URL/structure features
- Balanced logistic regression; serialized with joblib

# Feature engineering
- Text: word and character n-grams capture vocabulary, misspellings, and obfuscation patterns
- URLs: count, total length, IP-host URLs, shorteners, and suspicious terms
- Structure: HTML/link markers, urgency terms, punctuation, currency, digits, uppercase, @, and period counts
- The feature blocks are concatenated before classification

# Benchmark performance
- Accuracy: 96.25% on 800 held-out emails
- Safe: precision 0.98, recall 0.95, F1 0.97
- Phishing: precision 0.93, recall 0.97, F1 0.95
- Confusion matrix: TN 461, FP 22, FN 8, TP 309
- Include artifact: artifacts/confusion_matrix.png

# Real-time FastAPI service
- GET /health verifies artifact availability
- POST /scan accepts {"text": "..."} with Pydantic validation and a 200,000-character limit
- Response: label, confidence, phishing_probability, model identifier
- OpenAPI documentation at /docs; run with uvicorn api:app --host 0.0.0.0 --port 8000

# Adversarial robustness probe
- Hand-curated 8-case diagnostic set: Cyrillic homograph, zero-width insertion, defanged URL, IP URL, shortener, HTML link, and two benign controls
- Result: 7/8 correct, 87.50% diagnostic accuracy
- All six phishing-style cases were detected; one benign newsletter was falsely flagged
- Robustness gaps motivate Unicode confusable analysis, URL canonicalization, reputation checks, and human review

# Deployment and next steps
- Load only trusted, version-pinned model artifacts; joblib loading is not safe for untrusted files
- Place API behind TLS, authentication, rate limiting, monitoring, and a production ASGI process manager
- Add time-based and organization-specific validation, email headers/authentication, redirect analysis in a sandbox, and threshold tuning
- References: FastAPI request bodies and deployment docs; Scikit-learn model persistence docs; Hugging Face dataset card
