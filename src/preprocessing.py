"""Reusable text preprocessing for sentiment analysis (NLTK).

Steps: normalise -> tokenise -> POS-tag -> remove stop words (keeping negations)
-> POS-guided WordNet lemmatisation -> join tokens back into one string.

The same ``TextPreprocessor`` is the first step of the saved scikit-learn
pipeline, so training, evaluation and inference share one implementation.
The transformer is stateless (it learns nothing from data), so it cannot leak
information from the test set.
"""
from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Negations flip sentiment ("not good"), so they must survive stop-word removal.
NEGATIONS = frozenset(
    {"no", "nor", "not", "never", "neither", "none", "nobody", "nothing",
     "nowhere", "cannot", "without"}
)
# Words that modulate sentiment strength or contrast; kept deliberately.
EXTRA_KEEP = frozenset({"very", "too", "but"})
KEEP_WORDS = NEGATIONS | EXTRA_KEEP

# (path used by nltk.data.find, package name for nltk.download).
# Both the old ("punkt", "averaged_perceptron_tagger") and the newer
# ("punkt_tab", "averaged_perceptron_tagger_eng") names are handled because
# NLTK changed resource names across versions.
REQUIRED_NLTK_RESOURCES = (
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
)

_NLTK_READY = False

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_HTML_RE = re.compile(r"<[^>]+>")
_ALNUM_RE = re.compile(r"[a-z0-9]")
_CONTRACTIONS = (
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"n't\b"), " not"),
)


def ensure_nltk_resources(quiet: bool = True) -> None:
    """Make sure every NLTK resource needed by this module is available.

    Downloads missing resources, then verifies the real operations work and
    raises a RuntimeError with manual instructions if they do not.
    """
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    for path, package in REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=quiet)
            except Exception:  # offline, or package absent in this NLTK version
                pass
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        nltk.word_tokenize("A quick test.")
        nltk.pos_tag(["test"])
        stopwords.words("english")
        WordNetLemmatizer().lemmatize("tests")
    except Exception as exc:
        raise RuntimeError(
            "NLTK resources are missing or could not be downloaded "
            f"({type(exc).__name__}: {exc}).\n"
            "Connect to the internet and run:\n"
            "  python -m nltk.downloader punkt_tab punkt stopwords wordnet "
            "averaged_perceptron_tagger_eng averaged_perceptron_tagger"
        ) from exc
    _NLTK_READY = True


def _wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank tag to a WordNet POS letter (default: noun)."""
    if treebank_tag.startswith("J"):
        return "a"
    if treebank_tag.startswith("V"):
        return "v"
    if treebank_tag.startswith("R"):
        return "r"
    return "n"


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: iterable of raw reviews -> list of cleaned strings."""

    def __init__(self, lemmatize: bool = True, remove_stopwords: bool = True):
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords

    # --- scikit-learn API -------------------------------------------------
    def fit(self, X: Iterable, y=None) -> "TextPreprocessor":
        return self

    def transform(self, X: Iterable) -> List[str]:
        return [self.preprocess_text(text) for text in X]

    # --- serialisation: never pickle NLTK objects --------------------------
    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_stop_words", None)
        state.pop("_lemmatizer", None)
        return state

    # --- core logic -------------------------------------------------------
    def _load_resources(self) -> None:
        if getattr(self, "_lemmatizer", None) is not None:
            return
        ensure_nltk_resources()
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        self._stop_words = set(stopwords.words("english")) - KEEP_WORDS
        self._lemmatizer = WordNetLemmatizer()

    @staticmethod
    def normalize(text: str) -> str:
        """Lower-case, strip HTML/URLs, expand negative contractions."""
        text = text.replace("\u2019", "'").replace("\u2018", "'").lower()
        text = _HTML_RE.sub(" ", text)
        text = _URL_RE.sub(" ", text)
        for pattern, replacement in _CONTRACTIONS:
            text = pattern.sub(replacement, text)
        return re.sub(r"\s+", " ", text).strip()

    def preprocess_text(self, text) -> str:
        """Clean one review. Missing or empty input returns an empty string."""
        if text is None or (not isinstance(text, str) and pd.isna(text)):
            return ""
        if not isinstance(text, str):
            text = str(text)
        text = self.normalize(text)
        if not text:
            return ""

        import nltk

        self._load_resources()
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if _ALNUM_RE.search(t)]  # drop pure punctuation
        if not tokens:
            return ""
        tagged = nltk.pos_tag(tokens)

        out: List[str] = []
        for token, tag in tagged:
            if self.remove_stopwords and token in self._stop_words:
                continue
            if self.lemmatize:
                token = self._lemmatizer.lemmatize(token, pos=_wordnet_pos(tag))
            out.append(token)
        return " ".join(out)
