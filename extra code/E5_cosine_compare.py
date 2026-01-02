# 6_cosine_compare.py (concise)
# Compare consecutive cosine similarity from Count vs TF-IDF.
# Input : data_clean/ecb_statements_preprocessed.csv
# Output:
#   - data_features/ecb_similarity_cosines.csv
#   - data_features/cosine_series.png
#   - data_features/cosine_scatter.png

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CONFIG = {
    "INPUT_CSV": "data_clean/ecb_statements_preprocessed.csv",
    "OUTPUT_CSV": "data_features/ecb_similarity_cosines.csv",
    "TEXT_COL": "stems_str",          # "stems_str" or "tokens_clean_str"
    "NGRAM_RANGE": (1, 2),
    "MIN_DF": 1,
    "MAX_DF": 0.95,                  # None to disable
    "EPS": 1e-6,
    "DATE_MIN": None,                # e.g. "1999-01-01"
    "DATE_MAX": None,                # e.g. "2013-12-31"
}


def consecutive_cosine(X) -> np.ndarray:
    n = X.shape[0]
    sim = np.full(n, np.nan, dtype=float)
    for t in range(1, n):
        sim[t] = float(cosine_similarity(X[t], X[t - 1])[0, 0])
    return sim


def main() -> None:
    # script lives in extra_code/, so project root is parent
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent

    in_path = project_root / CONFIG["INPUT_CSV"]
    out_path = project_root / CONFIG["OUTPUT_CSV"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_csv(in_path)
    if "date" not in df.columns:
        raise ValueError("Input missing 'date' column")
    if CONFIG["TEXT_COL"] not in df.columns:
        raise ValueError(f"Input missing '{CONFIG['TEXT_COL']}' column")

    df = df.copy()
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values("date_dt").reset_index(drop=True)

    if CONFIG["DATE_MIN"]:
        df = df[df["date_dt"] >= pd.to_datetime(CONFIG["DATE_MIN"])]
    if CONFIG["DATE_MAX"]:
        df = df[df["date_dt"] <= pd.to_datetime(CONFIG["DATE_MAX"])]
    df = df.reset_index(drop=True)

    texts = df[CONFIG["TEXT_COL"]].fillna("").astype(str).tolist()
    if len(texts) < 2:
        raise ValueError("Not enough documents to compute similarities (need >=2).")

    common_kwargs = dict(
        ngram_range=CONFIG["NGRAM_RANGE"],
        min_df=CONFIG["MIN_DF"],
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    if CONFIG["MAX_DF"] is not None:
        common_kwargs["max_df"] = CONFIG["MAX_DF"]

    X_count = CountVectorizer(**common_kwargs).fit_transform(texts)
    X_tfidf = TfidfVectorizer(**common_kwargs).fit_transform(texts)

    sim_count = consecutive_cosine(X_count)
    sim_tfidf = consecutive_cosine(X_tfidf)

    eps = float(CONFIG["EPS"])
    log_count = np.where(np.isnan(sim_count), np.nan, np.log(sim_count + eps))
    log_tfidf = np.where(np.isnan(sim_tfidf), np.nan, np.log(sim_tfidf + eps))

    out = pd.DataFrame({
        "date": df["date"],
        "url": df["url"] if "url" in df.columns else "",
        "sim_countcos": sim_count,
        "sim_tfidfcos": sim_tfidf,
        "log_sim_countcos": log_count,
        "log_sim_tfidfcos": log_tfidf,
        "text_col": CONFIG["TEXT_COL"],
        "ngram_range": [str(CONFIG["NGRAM_RANGE"])] * len(df),
    })
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")

    valid = out[["sim_countcos", "sim_tfidfcos"]].dropna()
    corr = valid["sim_countcos"].corr(valid["sim_tfidfcos"])
    diff = valid["sim_tfidfcos"] - valid["sim_countcos"]
    print(f"n_docs={len(df)} | n_pairs={len(valid)} | corr={corr:.3f}")
    print(f"diff(tfidf-count): mean={diff.mean():.4f} median={diff.median():.4f} std={diff.std(ddof=1):.4f}")

    # --- Figures ---
    plt.figure()
    plt.plot(df["date_dt"], out["sim_countcos"], label="Cosine (Counts)")
    plt.plot(df["date_dt"], out["sim_tfidfcos"], label="Cosine (TF-IDF)")
    plt.legend()
    plt.title("Consecutive cosine similarity over time")
    plt.xlabel("Date")
    plt.ylabel("Similarity")
    plt.tight_layout()
    fig_series = out_path.parent / "cosine_series.png"
    plt.savefig(fig_series, dpi=200)
    print(f"Saved figure: {fig_series}")

    plt.figure()
    plt.scatter(valid["sim_countcos"], valid["sim_tfidfcos"], s=10)
    plt.title("TF-IDF cosine vs Count cosine (consecutive)")
    plt.xlabel("Cosine (Counts)")
    plt.ylabel("Cosine (TF-IDF)")
    plt.tight_layout()
    fig_scatter = out_path.parent / "cosine_scatter.png"
    plt.savefig(fig_scatter, dpi=200)
    print(f"Saved figure: {fig_scatter}")


if __name__ == "__main__":
    main()
