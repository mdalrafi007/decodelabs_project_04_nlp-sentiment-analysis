"""Exploratory data analysis: summary statistics and saved figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in Termux without a display
import matplotlib.pyplot as plt
import pandas as pd

from .model import LABEL_NAMES

COLORS = {0: "#d9534f", 1: "#5cb85c"}


def run_eda(df: pd.DataFrame, fig_dir: Path) -> dict:
    """Compute EDA statistics and save three figures into ``fig_dir``."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["n_words"] = df["text"].str.split().str.len()
    df["n_chars"] = df["text"].str.len()

    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([LABEL_NAMES[i] for i in counts.index], counts.values,
           color=[COLORS[i] for i in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Sentiment class distribution")
    ax.set_ylabel("Number of reviews")
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["n_words"], bins=30, color="#337ab7", edgecolor="white")
    ax.set_title("Review length distribution")
    ax.set_xlabel("Words per review")
    ax.set_ylabel("Number of reviews")
    fig.tight_layout()
    fig.savefig(fig_dir / "review_length_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    groups = [df.loc[df["label"] == k, "n_words"] for k in sorted(LABEL_NAMES)]
    ax.boxplot(groups, tick_labels=[LABEL_NAMES[k] for k in sorted(LABEL_NAMES)])
    ax.set_title("Review length by sentiment")
    ax.set_ylabel("Words per review")
    fig.tight_layout()
    fig.savefig(fig_dir / "review_length_by_sentiment.png", dpi=150)
    plt.close(fig)

    by_class = df.groupby("label")["n_words"].describe().round(2)
    examples = {
        LABEL_NAMES[k]: df.loc[df["label"] == k, "text"].head(3).tolist()
        for k in sorted(LABEL_NAMES)
    }
    return {
        "n_reviews": int(len(df)),
        "columns": {c: str(t) for c, t in df[["text", "label"]].dtypes.items()},
        "class_counts": {LABEL_NAMES[k]: int(v) for k, v in counts.items()},
        "word_length_overall": df["n_words"].describe().round(2).to_dict(),
        "word_length_by_class": {LABEL_NAMES[k]: row.to_dict() for k, row in by_class.iterrows()},
        "char_length_mean": round(float(df["n_chars"].mean()), 2),
        "examples": examples,
    }
