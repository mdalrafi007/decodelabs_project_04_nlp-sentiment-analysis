"""Single source of truth for the model pipeline and hyper-parameter grids."""
from __future__ import annotations

from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocessing import TextPreprocessor

SEED = 42
LABEL_NAMES = {0: "Negative", 1: "Positive"}
MODEL_NAMES = ("naive_bayes", "linear_svm")
DISPLAY_NAMES = {"naive_bayes": "Multinomial Naive Bayes", "linear_svm": "Linear SVM"}


def build_pipeline(model_name: str, seed: int = SEED, memory: Optional[str] = None) -> Pipeline:
    """Preprocessing -> TF-IDF -> classifier, as one scikit-learn Pipeline.

    TF-IDF defaults: unigrams+bigrams (captures "not good"); max_features caps
    vocabulary size for memory; min_df=2 drops one-off typos; max_df=0.95 drops
    near-universal terms; sublinear_tf dampens repeated words.
    """
    tfidf = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
    )
    if model_name == "naive_bayes":
        clf = MultinomialNB()
    elif model_name == "linear_svm":
        clf = LinearSVC(random_state=seed, max_iter=10000)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {MODEL_NAMES}.")
    return Pipeline(
        [("prep", TextPreprocessor()), ("tfidf", tfidf), ("clf", clf)], memory=memory
    )


# Small grids so tuning stays quick on a phone. Tuned with cross-validation on
# the TRAINING split only; the test split is never used for selection.
PARAM_GRIDS = {
    "naive_bayes": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "clf__alpha": [0.1, 0.3, 1.0],
    },
    "linear_svm": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "clf__C": [0.1, 1.0, 10.0],
    },
}
