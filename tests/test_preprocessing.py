"""Preprocessing tests (need NLTK data; skipped with a message if unavailable)."""
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import TextPreprocessor, ensure_nltk_resources


@pytest.fixture(scope="module")
def prep():
    try:
        ensure_nltk_resources()
    except RuntimeError as exc:
        pytest.skip(f"NLTK resources unavailable: {exc}")
    return TextPreprocessor()


def test_normal_text_returns_string(prep):
    out = prep.preprocess_text("The battery life is great!")
    assert isinstance(out, str) and "battery" in out.split()


@pytest.mark.parametrize("value", ["", "   ", None, float("nan"), "!!!"])
def test_empty_and_missing_values_give_empty_string(prep, value):
    assert prep.preprocess_text(value) == ""


def test_negation_is_kept(prep):
    assert "not" in prep.preprocess_text("This is not good").split()
    assert "not" in prep.preprocess_text("It didn't work").split()
    assert "never" in prep.preprocess_text("I will never buy this again").split()


def test_stop_words_removed(prep):
    tokens = prep.preprocess_text("the phone is on the table").split()
    assert "the" not in tokens and "is" not in tokens


def test_lemmatization(prep):
    assert "battery" in prep.preprocess_text("The batteries are great").split()


def test_output_is_tfidf_compatible(prep):
    docs = prep.transform(["Great phone", "Terrible phone, not good"])
    matrix = TfidfVectorizer().fit_transform(docs)
    assert matrix.shape[0] == 2 and matrix.shape[1] > 0
