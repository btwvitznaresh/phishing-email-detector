# Implementation references

1. FastAPI request bodies are declared with Pydantic models, which provides structured validation for JSON payloads: https://fastapi.tiangolo.com/tutorial/body/
2. FastAPI deployment guidance: https://fastapi.tiangolo.com/deployment/
3. Scikit-learn model persistence: joblib is efficient for large NumPy-backed models, but pickle/joblib artifacts must only be loaded from trusted, verified sources because loading can execute arbitrary code. ONNX or a sandboxed serving environment is recommended when appropriate: https://scikit-learn.org/stable/model_persistence.html
4. The public training corpus used by this project: https://huggingface.co/datasets/zefang-liu/phishing-email-dataset

Design implications: the API should validate bounded input lengths, load the locally built artifact only, avoid logging raw email bodies, and be deployed behind TLS, authentication/rate limiting, and a production ASGI server such as Uvicorn/Gunicorn.
