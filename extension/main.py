# E5_tfidf_cosine.py
# Compute consecutive cosine similarity using TF-IDF representations.
# Input : data_clean/ecb_statements_preprocessed.csv
# Output: data_features/ecb_similarity_tfidf.csv
# (+ time series plot saved to outputs/plots/ts_sim_tfidf.png)

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")  # for headless environments (server/CI)
import matplotlib.pyplot as plt

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

    # Optional: restrict to paper-like sample (used if env vars not set)
    "DATE_MIN": "1999-01-01",
    "DATE_MAX": "2013-12-31",

    # Plot output dir
    "PLOTS_DIR": "outputs/plots",
}


def get_dates() -> tuple[str | None, str | None]:
    # extension/main.py injects ECB_START_DATE / ECB_END_DATE
    dmin = os.getenv("ECB_START_DATE") or CONFIG.get("DATE_MIN")
    dmax = os.getenv("ECB_END_DATE") or CONFIG.get("DATE_MAX")
    return dmin, dmax


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
    date_min, date_max = get_dates()
    if date_min:
        df = df[df["date_dt"] >= pd.to_datetime(date_min)]
    if date_max:
        df = df[df["date_dt"] <= pd.to_datetime(date_max)]
    df = df.reset_index(drop=True)

    print(f"Date filter: {date_min=} {date_max=} | n_docs={len(df)}")

    texts = df[CONFIG["TEXT_COL"]].fillna("").astype(str).tolist()
    if len(texts) < 2:
        raise ValueError(
            f"Not enough documents to compute consecutive similarities (n_docs={len(texts)}). "
            f"Check date window: {date_min=} {date_max=}"
        )

    # TF-IDF vectorization (fit on the full corpus!)
    kwargs = dict(
        ngram_range=CONFIG["NGRAM_RANGE"],
        min_df=CONFIG["MIN_DF"],
        lowercase=False,              # already lowercased by preprocessing
        token_pattern=r"(?u)\b\w+\b", # keep short stems
    )
    if CONFIG["MAX_DF"] is not None:
        kwargs["max_df"] = CONFIG["MAX_DF"]

    X = TfidfVectorizer(**kwargs).fit_transform(texts)  # (n_docs, n_features)

    # Consecutive cosine similarity: sim[t] = cos(X[t], X[t-1])
    sim = np.full(len(df), np.nan, dtype=float)
    for t in range(1, len(df)):
        sim[t] = float(cosine_similarity(X[t], X[t - 1])[0, 0])

    # Log transform (optional, for regressions like the paper)
    eps = float(CONFIG["EPS"])
    log_sim = np.full_like(sim, np.nan)
    for i, v in enumerate(sim):
        if not np.isnan(v):
            log_sim[i] = math.log(v + eps)

    out = pd.DataFrame({
        "date": df["date"],
        "url": df["url"] if "url" in df.columns else "",
        "sim_tfidf": sim,
        "log_sim_tfidf": log_sim,
        "text_col": CONFIG["TEXT_COL"],
        "ngram_range": str(CONFIG["NGRAM_RANGE"]),
    })

    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")

    # Plot: time series sim_tfidf
    plots_dir = project_root / CONFIG["PLOTS_DIR"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 4.5))
    plt.plot(df["date_dt"], pd.to_numeric(out["sim_tfidf"], errors="coerce"))
    plt.title("Consecutive TF-IDF cosine similarity (sim_tfidf)")
    plt.xlabel("Date")
    plt.ylabel("sim_tfidf")
    plt.tight_layout()

    p_ts = plots_dir / "ts_sim_tfidf.png"
    fig.savefig(p_ts, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {p_ts}")

    valid = out["sim_tfidf"].dropna()
    print(f"n_docs={len(df)} | n_sims={len(valid)}")
    if len(valid) > 0:
        print(f"sim_tfidf min={valid.min():.4f} mean={valid.mean():.4f} max={valid.max():.4f}")


if __name__ == "__main__":
    main()
