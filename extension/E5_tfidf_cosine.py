# E5_tfidf_cosine.py
# Compute consecutive cosine similarity using TF-IDF representations.
# Input : data_clean/ecb_statements_preprocessed.csv
# Output: data_features/ecb_similarity_tfidf.csv

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CONFIG = {
    "INPUT_CSV": "data_clean/ecb_statements_preprocessed.csv",
    "OUTPUT_CSV": "data_features/ecb_similarity_tfidf.csv",

    # We choose: "stems_str" (recommended to match the paper) or "tokens_clean_str"
    "TEXT_COL": "stems_str",

    # TF-IDF setup
    "NGRAM_RANGE": (1, 2),
    "MIN_DF": 1,
    "MAX_DF": 0.95,   # set None to disable

    # For log(sim) to avoid log(0)
    "EPS": 1e-6,

    # Optional: restrict to paper-like sample
    "DATE_MIN": None,  # e.g. "1999-01-01"
    "DATE_MAX": None,  # e.g. "2013-12-31"
}


def main() -> None:
    # Script is in extension/, so project root is one level above.
    scripts_dir = Path(__file__).resolve().parent      # .../extension
    project_root = scripts_dir.parent                  # .../ (repo root)

    in_path = project_root / CONFIG["INPUT_CSV"]
    out_path = project_root / CONFIG["OUTPUT_CSV"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input introuvable: {in_path} (run replication preprocess step first)")

    df = pd.read_csv(in_path)

    if "date" not in df.columns:
        raise ValueError("Input missing 'date' column")
    if CONFIG["TEXT_COL"] not in df.columns:
        raise ValueError(f"Input missing '{CONFIG['TEXT_COL']}' column")

    # Parse + sort by date
    df = df.copy()
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"].notna()].sort_values("date_dt").reset_index(drop=True)

    # Optional date filter
    if CONFIG["DATE_MIN"]:
        df = df[df["date_dt"] >= pd.to_datetime(CONFIG["DATE_MIN"])].copy()
    if CONFIG["DATE_MAX"]:
        df = df[df["date_dt"] <= pd.to_datetime(CONFIG["DATE_MAX"])].copy()
    df = df.reset_index(drop=True)

    texts = df[CONFIG["TEXT_COL"]].fillna("").astype(str).tolist()
    if len(texts) < 2:
        raise ValueError("Not enough documents to compute consecutive similarities.")

    # TF-IDF vectorization (fit on the full corpus!)
    kwargs = dict(
        ngram_range=CONFIG["NGRAM_RANGE"],
        min_df=CONFIG["MIN_DF"],
        lowercase=False,              # already lowercased by preprocessing
        token_pattern=r"(?u)\b\w+\b", # keep short stems
    )
    if CONFIG["MAX_DF"] is not None:
        kwargs["max_df"] = CONFIG["MAX_DF"]

    vec = TfidfVectorizer(**kwargs)
    X = vec.fit_transform(texts)  # (n_docs, n_features)

    # Consecutive cosine similarity: sim[t] = cos(X[t], X[t-1])
    sim = np.full(shape=(len(df),), fill_value=np.nan, dtype=float)
    for t in range(1, len(df)):
        sim[t] = float(cosine_similarity(X[t], X[t - 1])[0, 0])

    # Log transform (optional, for regressions like the paper)
    eps = float(CONFIG["EPS"])
    log_sim = np.full_like(sim, np.nan)
    for i in range(len(sim)):
        if not np.isnan(sim[i]):
            log_sim[i] = math.log(sim[i] + eps)

    out = pd.DataFrame({
        "date": df["date"].astype(str).str.slice(0, 10),
        "url": df["url"] if "url" in df.columns else "",
        "sim_tfidf": sim,
        "log_sim_tfidf": log_sim,
        "text_col": CONFIG["TEXT_COL"],
        "ngram_range": [str(CONFIG["NGRAM_RANGE"])] * len(df),
    })

    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")

    valid = out["sim_tfidf"].dropna()
    print(f"n_docs={len(df)} | n_sims={len(valid)}")
    print(f"sim_tfidf min={valid.min():.4f} mean={valid.mean():.4f} max={valid.max():.4f}")


if __name__ == "__main__":
    main()
