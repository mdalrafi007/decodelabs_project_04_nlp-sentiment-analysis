"""Dataset acquisition, loading and cleaning (UCI Sentiment Labelled Sentences)."""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
UCI_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
SUBSET_FILES: Dict[str, str] = {
    "amazon": "amazon_cells_labelled.txt",
    "imdb": "imdb_labelled.txt",
    "yelp": "yelp_labelled.txt",
}


class DatasetUnavailableError(RuntimeError):
    """Raised when the dataset cannot be downloaded or found."""


def _manual_instructions() -> str:
    return (
        "Could not obtain the dataset automatically.\n"
        "Manual setup:\n"
        "  1. Open https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences\n"
        "  2. Download the zip and extract 'amazon_cells_labelled.txt'.\n"
        f"  3. Place it at: {RAW_DIR / SUBSET_FILES['amazon']}\n"
        "Then re-run the command."
    )


def download_dataset(subsets: Optional[List[str]] = None) -> None:
    """Download and extract the requested subset files into data/raw/ if absent."""
    subsets = subsets or ["amazon"]
    wanted = {SUBSET_FILES[s] for s in subsets}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DIR / name).exists() for name in wanted):
        return
    zip_path = RAW_DIR / "sentiment_labelled_sentences.zip"
    try:
        urllib.request.urlretrieve(UCI_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                base = Path(member).name
                if base in wanted and "__MACOSX" not in member:
                    (RAW_DIR / base).write_bytes(archive.read(member))
    except Exception as exc:
        raise DatasetUnavailableError(f"{exc}\n{_manual_instructions()}") from exc
    missing = [n for n in wanted if not (RAW_DIR / n).exists()]
    if missing:
        raise DatasetUnavailableError(f"Archive lacked {missing}.\n{_manual_instructions()}")


def _read_labelled_file(path: Path) -> Tuple[pd.DataFrame, int]:
    """Parse 'sentence<TAB>label' lines. Returns (frame, malformed_line_count)."""
    if not path.exists():
        raise DatasetUnavailableError(f"File not found: {path}\n{_manual_instructions()}")
    texts, labels, malformed = [], [], 0
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) != 2 or parts[1].strip() not in {"0", "1"}:
                malformed += 1
                continue
            texts.append(parts[0])
            labels.append(int(parts[1].strip()))
    return pd.DataFrame({"text": texts, "label": labels}), malformed


def load_reviews(subset: str = "amazon", path: Optional[Path] = None) -> Tuple[pd.DataFrame, int]:
    """Load raw reviews. Labels: 0 = Negative, 1 = Positive (as in the source)."""
    if path is not None:
        df, bad = _read_labelled_file(Path(path))
        df["source"] = Path(path).stem
        return df, bad
    names = list(SUBSET_FILES) if subset == "all" else [subset]
    if any(n not in SUBSET_FILES for n in names):
        raise ValueError(f"subset must be one of {list(SUBSET_FILES)} or 'all'")
    frames, total_bad = [], 0
    for name in names:
        df, bad = _read_labelled_file(RAW_DIR / SUBSET_FILES[name])
        df["source"] = name
        frames.append(df)
        total_bad += bad
    return pd.concat(frames, ignore_index=True), total_bad


def clean_reviews(df: pd.DataFrame, malformed: int = 0) -> Tuple[pd.DataFrame, dict]:
    """Drop empty, conflicting and duplicate rows; return (clean_df, report)."""
    report = {"rows_loaded": int(len(df)), "malformed_lines_skipped": int(malformed)}
    df = df.copy()
    df["text"] = df["text"].where(df["text"].notna(), "").astype(str).str.strip()

    empty = df["text"] == ""
    report["missing_or_empty_text"] = int(empty.sum())
    df = df[~empty]

    label_counts = df.groupby("text")["label"].nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    report["texts_with_conflicting_labels_dropped"] = len(conflicting)
    df = df[~df["text"].isin(conflicting)]

    dup = df.duplicated(subset=["text", "label"])
    report["exact_duplicates_dropped"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    report["rows_after_cleaning"] = int(len(df))
    report["class_counts"] = {
        "negative_0": int((df["label"] == 0).sum()),
        "positive_1": int((df["label"] == 1).sum()),
    }
    return df, report
