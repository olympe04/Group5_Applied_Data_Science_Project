# E10_information_structure_features.py

from __future__ import annotations

from collections import Counter
from math import log
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CFG = {
    "IN_FILE": "data_clean/ecb_statements_preprocessed.csv",
    "OUT_FILE": "data_features/ecb_information_structure.csv",
    "TEXT_COL": "tokens_clean_str",
    "PLOT_DIR": "outputs/plots",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_preprocessed(path: Path, text_col: str) -> pd.DataFrame:
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
    d = df.copy()
    d["len_for_dedupe"] = d[text_col].apply(lambda s: len(s.split()))
    return (
        d.sort_values(["date", "len_for_dedupe"], ascending=[True, False])
        .drop_duplicates(subset=["date"], keep="first")
        .sort_values("date")
        .reset_index(drop=True)
    )


def shannon_entropy(tokens: list[str]) -> float | None:
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
    rows = []
    historical_vocab: set[str] = set()

    for _, r in df.iterrows():
        tokens = r[text_col].split()
        n = len(tokens)
        types = set(tokens)
        v = len(types)

        h = shannon_entropy(tokens)
        if h is None or v == 0:
            h_norm = None
        elif v == 1:
            h_norm = 0.0
        else:
            h_norm = h / log(v)

        ttr = (v / n) if n > 0 else None
        herdan = (log(v) / log(n)) if (n > 1 and v > 0) else None

        bigrams = list(zip(tokens, tokens[1:])) if n >= 2 else []
        bigrams_u = len(set(bigrams))
        bigram_unique_ratio = (bigrams_u / (n - 1)) if n >= 2 else None

        ratio_new_types = (len(types - historical_vocab) / v) if v > 0 else None
        ratio_new_tokens = (sum(1 for w in tokens if w not in historical_vocab) / n) if n > 0 else None

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
    root = get_project_root()
    in_path = root / CFG["IN_FILE"]
    out_path = root / CFG["OUT_FILE"]
    plot_dir = root / CFG["PLOT_DIR"]

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = load_preprocessed(in_path, CFG["TEXT_COL"])
    df = dedupe_by_date_keep_longest(df, CFG["TEXT_COL"])
    feats = compute_features(df, CFG["TEXT_COL"])

    # Print full descriptive table for ALL columns (including date)
    print(feats.describe(include="all").T.to_string())

    # Save plots to outputs/plots
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

    # Save CSV (date formatted for downstream merges)
    feats_out = feats.copy()
    feats_out["date"] = feats_out["date"].dt.strftime("%Y-%m-%d")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats_out.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nSaved: {out_path} | n={len(feats_out)}")
    print(f"Saved plots in: {plot_dir}")


if __name__ == "__main__":
    main()
