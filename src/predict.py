"""Predict the sentiment of a new review with the saved model.

Run from the project root:
    python -m src.predict "Great battery life, works perfectly"
    python -m src.predict --demo
    python -m src.predict            # interactive prompt
Only load .joblib files you created yourself (pickles can run arbitrary code).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import joblib

from .model import LABEL_NAMES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sentiment_model.joblib"

# Illustrative inputs written for this demo. They are NOT evaluation data.
DEMO_REVIEWS = [
    "Great battery life, I love this phone.",
    "Terrible quality, it broke after two days.",
    "This is not good at all.",
    "Not bad, actually works fine.",
    "It's okay I guess.",
    "The charger arrived on Tuesday.",
]


def load_model(path: Optional[Path] = None):
    """Load the saved pipeline (preprocessing + TF-IDF + classifier)."""
    path = Path(path) if path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def predict_review(text: str, model=None, model_path: Optional[Path] = None) -> dict:
    """Return the predicted label for one review.

    Raises ValueError for non-string, empty, or no-usable-word input.
    The result includes ``probability_positive`` only for models that provide
    probabilities (Naive Bayes, whose estimates can be poorly calibrated);
    for the linear SVM it gives ``decision_score``, which is NOT a probability.
    """
    if not isinstance(text, str):
        raise ValueError("Review must be a string.")
    if not text.strip():
        raise ValueError("Review is empty.")
    model = model if model is not None else load_model(model_path)
    if not model.named_steps["prep"].preprocess_text(text):
        raise ValueError("No usable words found in the review after preprocessing.")

    label_id = int(model.predict([text])[0])
    result = {"text": text, "label": LABEL_NAMES[label_id]}
    if hasattr(model, "predict_proba"):
        result["probability_positive"] = float(model.predict_proba([text])[0][1])
    elif hasattr(model, "decision_function"):
        result["decision_score"] = float(model.decision_function([text])[0])
    return result


def _format(result: dict) -> str:
    extra = ""
    if "probability_positive" in result:
        extra = f"  (P(Positive) ~ {result['probability_positive']:.2f}, may be poorly calibrated)"
    elif "decision_score" in result:
        extra = f"  (SVM decision score {result['decision_score']:+.2f}; > 0 means Positive, not a probability)"
    return f"{result['label']}{extra}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict review sentiment.")
    parser.add_argument("text", nargs="*", help="Review text.")
    parser.add_argument("--demo", action="store_true", help="Run illustrative examples.")
    parser.add_argument("--model", default=None, help="Path to a saved .joblib model.")
    args = parser.parse_args()
    try:
        model = load_model(args.model)
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")

    def show(review: str) -> None:
        try:
            print(f"> {review}\n  {_format(predict_review(review, model=model))}")
        except ValueError as exc:
            print(f"> {review}\n  Cannot predict: {exc}")

    if args.demo:
        print("Illustrative examples (not from the evaluation set):")
        for review in DEMO_REVIEWS:
            show(review)
    elif args.text:
        show(" ".join(args.text))
    else:
        print("Type a review and press Enter (empty line or 'quit' to exit).")
        while True:
            try:
                line = input("review> ").strip()
            except EOFError:
                break
            if line.lower() in {"", "quit", "exit"}:
                break
            show(line)


if __name__ == "__main__":
    main()
