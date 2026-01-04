# E10_information_structure_features.py
#
# Build "information structure" / text-complexity features from ECB statements.
# The idea: capture how diverse / informative the language is over time and how much "new"
# vocabulary appears relative to all previously seen statements.
#
# Features computed per event/date (after deduping to 1 statement per date):
#   - n_tokens: number of word tokens
#   - v_types: number of unique word types (vocabulary size)
#   - ttr: type-token ratio = v_types / n_tokens
#   - herdan_c: Herdan's C = log(v_types) / log(n_tokens) (lexical richness, length-adjusted)
#   - entropy: Shannon entropy of token distribution (natural log)
#   - entropy_norm: entropy normalized by log(v_types) in [0,1] if v_types>1
#   - bigrams_unique: number of unique bigrams
#   - bigrams_unique_ratio: unique bigrams / (n_tokens-1)
#   - ratio_new_types: share of types not seen in any earlier statement
#   - ratio_new_tokens: share of tokens not seen in any earlier statement
#
# I/O:
#   Input : data_clean/ecb_statements_preprocessed.csv (must contain date + TEXT_COL)
#   Output: data_features/ecb_information_structure.csv
#   Plots : outputs/plots/entropy_over_time.png
#           outputs/plots/ratio_new_tokens_over_time.png

from __future__ import annotations

from collections import Counter
from math import log
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CFG = {
    "IN_FILE": "data_clean/ecb_statements_preprocessed.csv",
    "OUT_FILE": "data_features/ecb_information_structure.csv",
    "TEXT_COL": "tokens_clean_str",  # whitespace tokenized string of cleaned tokens
    "PLOT_DIR": "outputs/plots",
}


def get_project_root() -> Path:
    """
    Return the repository root so all relative paths resolve consistently.

    Assumption here: this script lives at:
      <root>/extension/<something>/E10_information_structure_features.py
    so going up 3 parents gets back to <root>.
    """
    return Path(__file__).resolve().parent.parent.parent


def load_preprocessed(path: Path, text_col: str) -> pd.DataFrame:
    """
    Load the preprocessed ECB statements dataset and enforce basic schema:
      - requires a parseable 'date' column
      - requires the chosen text column (e.g., tokens_clean_str)
    Returns a clean DataFrame with:
      - date as datetime
      - text_col as non-null string
    """
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {path}")
    if text_col not in df.columns:
        raise ValueError(f"Missing '{text_col}' column in {path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    return df


def dedupe_by_date_keep_longest(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Ensure ONE statement per calendar date.

    Some dates can have multiple scraped statements (duplicates, revisions, etc.).
    Strategy:
      - compute token length per row
      - sort by date then length (descending)
      - keep the longest statement per date
    """
    d = df.copy()
    d["len_for_dedupe"] = d[text_col].apply(lambda s: len(s.split()))
    return (
        d.sort_values(["date", "len_for_dedupe"], ascending=[True, False])
        .drop_duplicates(subset=["date"], keep="first")
        .sort_values("date")
        .reset_index(drop=True)
    )


def shannon_entropy(tokens: list[str]) -> float | None:
    """
    Compute Shannon entropy H(tokens) = - Σ p(w) log p(w) using natural logs.

    Interpretation:
      - higher entropy => token distribution is more diverse / less repetitive
      - None if there are no tokens
    """
    if not tokens:
        return None

    c = Counter(tokens)
    n = len(tokens)

    h = 0.0
    for freq in c.values():
        p = freq / n
        h -= p * log(p)
    return h


def compute_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Compute per-date information-structure features.

    We also maintain a cumulative historical vocabulary (set of all types seen so far)
    to quantify "novelty" for each new statement:
      - ratio_new_types  = |types \ historical_vocab| / |types|
      - ratio_new_tokens = (# tokens not in historical_vocab) / n_tokens
    """
    rows = []
    historical_vocab: set[str] = set()

    for _, r in df.iterrows():
        # --- basic tokenization (expects whitespace-separated tokens in the input col) ---
        tokens = r[text_col].split()
        n = len(tokens)
        types = set(tokens)
        v = len(types)

        # --- entropy and normalized entropy ---
        h = shannon_entropy(tokens)
        if h is None or v == 0:
            h_norm = None
        elif v == 1:
            # only one unique word => perfectly concentrated distribution
            h_norm = 0.0
        else:
            # normalize by maximum possible entropy log(v)
            h_norm = h / log(v)

        # --- lexical richness / length-adjusted metrics ---
        ttr = (v / n) if n > 0 else None
        herdan = (log(v) / log(n)) if (n > 1 and v > 0) else None

        # --- bigram diversity: unique bigrams / total bigram positions ---
        bigrams = list(zip(tokens, tokens[1:])) if n >= 2 else []
        bigrams_u = len(set(bigrams))
        bigram_unique_ratio = (bigrams_u / (n - 1)) if n >= 2 else None

        # --- novelty vs previously observed vocabulary ---
        ratio_new_types = (len(types - historical_vocab) / v) if v > 0 else None
        ratio_new_tokens = (sum(1 for w in tokens if w not in historical_vocab) / n) if n > 0 else None

        # update historical vocabulary AFTER computing novelty for this row
        historical_vocab |= types

        rows.append(
            {
                "date": r["date"],
                "n_tokens": n,
                "v_types": v,
                "ttr": ttr,
                "herdan_c": herdan,
                "entropy": h,
                "entropy_norm": h_norm,
                "bigrams_unique": bigrams_u,
                "bigrams_unique_ratio": bigram_unique_ratio,
                "ratio_new_types": ratio_new_types,
                "ratio_new_tokens": ratio_new_tokens,
            }
        )

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def save_time_plot(df: pd.DataFrame, y: str, out_png: Path, title: str) -> None:
    """
    Save a simple time-series line plot for column y against date.

    - drops missing values in y and date
    - sorts chronologically
    - writes a PNG to out_png
    """
    d = df[["date", y]].dropna().sort_values("date")
    plt.figure()
    plt.plot(d["date"], d[y])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    """
    Main runner:
      1) resolve paths from repo root
      2) load + validate input
      3) dedupe to one statement per date
      4) compute features
      5) print descriptive stats for sanity checks
      6) save a couple diagnostic plots
      7) export the feature dataset for downstream merges/regressions
    """
    root = get_project_root()
    in_path = root / CFG["IN_FILE"]
    out_path = root / CFG["OUT_FILE"]
    plot_dir = root / CFG["PLOT_DIR"]

    # --- input checks ---
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    # --- load + dedupe ---
    df = load_preprocessed(in_path, CFG["TEXT_COL"])
    df = dedupe_by_date_keep_longest(df, CFG["TEXT_COL"])

    # --- compute features ---
    feats = compute_features(df, CFG["TEXT_COL"])

    # Print full descriptive table for ALL columns (including date)
    print(feats.describe(include="all").T.to_string())

    # --- plots (diagnostics) ---
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_time_plot(
        feats,
        y="entropy",
        out_png=plot_dir / "entropy_over_time.png",
        title="ECB statement entropy over time",
    )
    save_time_plot(
        feats,
        y="ratio_new_tokens",
        out_png=plot_dir / "ratio_new_tokens_over_time.png",
        title="ECB statement new-token ratio over time",
    )

    # --- export CSV (date formatted for downstream merges) ---
    feats_out = feats.copy()
    feats_out["date"] = feats_out["date"].dt.strftime("%Y-%m-%d")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats_out.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nSaved: {out_path} | n={len(feats_out)}")
    print(f"Saved plots in: {plot_dir}")


if __name__ == "__main__":
    main()
