# extra_code/wordcloud_ecb_window.py
# Build a word cloud from ECB preprocessed tokens within an env window.
#
# I/O:
#   Input : data_clean/ecb_statements_preprocessed.csv (requires: date, tokens_clean_str)
#   Output: outputs/plots/wordcloud_ecb_<START>_<END>.png
#
# Usage:
#   ECB_START_DATE=1999-01-01 ECB_END_DATE=2013-12-31 python extra_code/wordcloud_ecb_window.py

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


DEFAULT_START = "1999-01-01"
DEFAULT_END = "2013-12-31"


def get_project_root() -> Path:
    # script in: project_root/extra_code/...
    return Path(__file__).resolve().parent.parent


def get_window() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    s = os.getenv("ECB_START_DATE", DEFAULT_START)
    e = os.getenv("ECB_END_DATE", DEFAULT_END)
    sdt, edt = pd.Timestamp(s), pd.Timestamp(e)
    if edt < sdt:
        raise ValueError(f"Invalid window: {s}..{e}")
    return sdt, edt, s, e


def main() -> None:
    root = get_project_root()
    sdt, edt, s, e = get_window()

    in_path = root / "data_clean" / "ecb_statements_preprocessed.csv"
    out_dir = root / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_csv(in_path)
    if "date" not in df.columns or "tokens_clean_str" not in df.columns:
        raise ValueError("Input must contain columns: 'date', 'tokens_clean_str'")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df[(df["date"] >= sdt) & (df["date"] <= edt)].copy()

    if len(df) == 0:
        raise ValueError(f"No rows in window {s}..{e}")

    tokens = []
    for txt in df["tokens_clean_str"].fillna("").astype(str):
        tokens.extend(txt.split())

    if not tokens:
        raise ValueError("No tokens found after filtering (tokens_clean_str is empty).")

    freq = Counter(tokens)

    wc = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        max_words=150,
        prefer_horizontal=0.9,
        collocations=False,
        colormap="plasma",
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()

    out_path = out_dir / f"wordcloud_ecb_{s.replace('-', '')}_{e.replace('-', '')}.png"
    wc.to_file(str(out_path))
    print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
