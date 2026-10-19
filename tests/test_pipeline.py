"""Pipeline tests on a tiny SYNTHETIC dataset (unit tests only, not evaluation)."""
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.model import MODEL_NAMES, build_pipeline
from src.predict import predict_review
from src.preprocessing import ensure_nltk_resources

ROOT = Path(__file__).resolve().parents[1]
POS = ["great phone love it", "excellent sound very happy", "works perfectly good quality",
       "amazing battery highly recommend", "fantastic product happy buyer", "love the great screen"]
NEG = ["terrible phone hate it", "awful sound very disappointed", "broke quickly bad quality",
       "horrible battery do not recommend", "worst product waste money", "hate the awful screen"]
TEXTS = POS * 2 + NEG * 2
LABELS = np.array([1] * 12 + [0] * 12)


@pytest.fixture(scope="module", autouse=True)
def _nltk():
    try:
        ensure_nltk_resources()
    except RuntimeError as exc:
        pytest.skip(f"NLTK resources unavailable: {exc}")


def _fit(name):
    pipe = build_pipeline(name)
    pipe.set_params(tfidf__min_df=1)
    return pipe.fit(TEXTS, LABELS)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fit_and_valid_labels(name):
    pipe = _fit(name)
    preds = pipe.predict(["great phone", "terrible awful phone", "something unseen entirely"])
    assert set(preds.tolist()) <= {0, 1}


def test_unseen_review_prediction_is_sensible():
    pipe = _fit("linear_svm")
    assert predict_review("love this great phone", model=pipe)["label"] == "Positive"
    assert predict_review("awful terrible hate it", model=pipe)["label"] == "Negative"


def test_invalid_inputs_rejected():
    pipe = _fit("naive_bayes")
    for bad in ["", "   ", "!!!", None, 123]:
        with pytest.raises(ValueError):
            predict_review(bad, model=pipe)


def test_tfidf_fitted_on_training_text_only():
    train_texts, test_texts = TEXTS[:20], ["zzzqqqonlytest great phone"]
    pipe = build_pipeline("naive_bayes")
    pipe.set_params(tfidf__min_df=1)
    pipe.fit(train_texts, LABELS[:20])
    pipe.predict(test_texts)  # transform only; must not alter the vocabulary
    assert not any("zzzqqqonlytest" in term for term in pipe.named_steps["tfidf"].vocabulary_)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_reproducible_with_fixed_seed(name):
    a, b = _fit(name), _fit(name)
    probe = ["great phone", "awful phone", "okay"]
    assert a.predict(probe).tolist() == b.predict(probe).tolist()


def test_saved_model_reloads_in_separate_process(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(_fit("linear_svm"), path)
    code = (
        "from src.predict import predict_review, load_model;"
        f"m = load_model(r'{path}');"
        "print(predict_review('great phone love it', model=m)['label'])"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() in {"Positive", "Negative"}
