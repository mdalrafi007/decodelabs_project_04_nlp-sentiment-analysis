"""Evaluation helpers: metrics, confusion matrices, comparison chart, features."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from .model import DISPLAY_NAMES, LABEL_NAMES

POSITIVE_CLASS = 1


def positive_scores(model, texts: Sequence[str]) -> np.ndarray:
    """Probability of Positive if available, otherwise the SVM decision score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)


def compute_metrics(model, X_test: Sequence[str], y_test: Sequence[int]) -> dict:
    """Metrics on held-out data. Positive class = 1 (Positive)."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, positive_scores(model, X_test))),
        "confusion_matrix": cm.tolist(),  # rows = true [Neg, Pos], cols = predicted
        "classification_report": classification_report(
            y_test, y_pred, target_names=[LABEL_NAMES[0], LABEL_NAMES[1]], zero_division=0
        ),
    }


def plot_confusion_matrix(cm: Sequence[Sequence[int]], title: str, path: Path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    labels = [LABEL_NAMES[0], LABEL_NAMES[1]]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(metrics_by_model: Dict[str, dict], path: Path) -> None:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    names = list(metrics_by_model)
    width = 0.8 / len(names)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, name in enumerate(names):
        vals = [metrics_by_model[name][k] for k in keys]
        ax.bar(x + i * width, vals, width, label=DISPLAY_NAMES.get(name, name))
    ax.set_xticks(x + width * (len(names) - 1) / 2, [k.upper() if k == "f1" else k for k in keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (held-out test set)")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_features(model, n: int = 15) -> pd.DataFrame:
    """Most Positive / Negative features of a fitted linear model.

    These are associations learned from this dataset, not facts about language.
    """
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    if hasattr(clf, "coef_"):
        weights = clf.coef_.ravel()  # positive weight -> class 1 (Positive)
    else:
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    order = np.argsort(weights)
    rows = [("Negative", names[i], float(weights[i])) for i in order[:n]]
    rows += [("Positive", names[i], float(weights[i])) for i in order[::-1][:n]]
    return pd.DataFrame(rows, columns=["direction", "feature", "weight"])


def error_table(texts: Sequence[str], y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    """All test reviews with correctness flags and error type."""
    df = pd.DataFrame({"text": list(texts), "actual": list(y_true), "predicted": list(y_pred)})
    df["correct"] = df["actual"] == df["predicted"]
    df["error_type"] = np.where(
        df["correct"], "",
        np.where(df["predicted"] == 1, "false_positive", "false_negative"),
    )
    df["actual"] = df["actual"].map(LABEL_NAMES)
    df["predicted"] = df["predicted"].map(LABEL_NAMES)
    return df
