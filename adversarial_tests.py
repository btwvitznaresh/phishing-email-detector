from __future__ import annotations

import json
from pathlib import Path

from predict_email import predict_email

CASES = [
    {"name": "homograph-paypal", "expected": "Phishing", "text": "Security alert: verify your PayPal account at https://раypal.example.invalid/login now."},
    {"name": "zero-width-login", "expected": "Phishing", "text": "Your account will be suspended. Please l\u200bog\u200bin at https://secure-login.example.invalid and confirm your password."},
    {"name": "defanged-url", "expected": "Phishing", "text": "Urgent invoice. Open hxxp://secure-login[.]example[.]invalid/verify and send your account details."},
    {"name": "ip-url", "expected": "Phishing", "text": "Microsoft 365 password expires today. Click http://192.0.2.10/office/verify immediately."},
    {"name": "shortened-url", "expected": "Phishing", "text": "You won a prize. Confirm your bank details now: https://bit.ly/example-test"},
    {"name": "html-link", "expected": "Phishing", "text": "<html><a href='https://account.example.invalid'>Click here</a> to verify your account immediately.</html>"},
    {"name": "benign-meeting", "expected": "Safe", "text": "Hi team, the project meeting is tomorrow at 3 PM. Please review the agenda and reply with comments."},
    {"name": "benign-newsletter", "expected": "Safe", "text": "This month’s library newsletter includes opening hours and upcoming public events. No response is required."},
]


def main():
    rows = []
    for case in CASES:
        label, confidence = predict_email(case["text"])
        rows.append({**case, "prediction": label, "confidence": confidence, "correct": label == case["expected"]})
    accuracy = sum(row["correct"] for row in rows) / len(rows)
    result = {"cases": rows, "accuracy": accuracy, "note": "Adversarial set is hand-curated and diagnostic, not a statistically representative benchmark."}
    Path("artifacts/adversarial_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in rows:
        print(f"{row['name']}: expected={row['expected']} predicted={row['prediction']} confidence={row['confidence']:.2%} correct={row['correct']}")
    print(f"Adversarial accuracy: {accuracy:.2%} ({sum(row['correct'] for row in rows)}/{len(rows)})")


if __name__ == "__main__":
    main()
