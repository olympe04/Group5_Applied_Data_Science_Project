# 6_cosine_compare.py (concise)
# Compare consecutive cosine similarity from Count vs TF-IDF.
# I/O:
#   Inputs : data_clean/ecb_statements_preprocessed.csv
#   Outputs: data_clean/ecb_similarity_cosines.csv,
#            outputs/plots/cosine_series.png, outputs/plots/cosine_scatter.png
# Notes:
#   The script builds Count and TF-IDF n-gram matrices on the same corpus, computes consecutive cosine similarity,
#   saves the series to CSV, and exports two diagnostic plots.

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
    "OUTPUT_CSV": "data_clean/ecb_similarity_cosines.csv",  # changed (was data_features/)
    "TEXT_COL": "stems_str",          # "stems_str" or "tokens_clean_str"
    "NGRAM_RANGE": (1, 2),
    "MIN_DF": 1,
    "MAX_DF": 0.95,                  # None to disable
    "EPS": 1e-6,
    "DATE_MIN": None,                # e.g. "1999-01-01"
    "DATE_MAX": None,                # e.g. "2013-12-31"
    "PLOTS_DIR": "outputs/plots",
}


def get_project_root() -> Path:
    """Return repository root (script is in extra_code/)."""
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir.parent


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path]:
    """Return (input_csv_path, output_csv_path, plots_dir) and create needed directories."""
    in_path = project_root / CONFIG["INPUT_CSV"]
    out_path = project_root / CONFIG["OUTPUT_CSV"]
    plots_dir = project_root / CONFIG["PLOTS_DIR"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return in_path, out_path, plots_dir


def load_and_filter(in_path: Path) -> pd.DataFrame:
    """Load input CSV, parse/sort dates, and apply optional date filters."""
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
    return df.reset_index(drop=True)


def make_vectorizers() -> dict:
    """Build shared vectorizer kwargs for Count and TF-IDF."""
    kwargs = dict(
        ngram_range=CONFIG["NGRAM_RANGE"],
        min_df=CONFIG["MIN_DF"],
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    if CONFIG["MAX_DF"] is not None:
        kwargs["max_df"] = CONFIG["MAX_DF"]
    return kwargs


def consecutive_cosine(X) -> np.ndarray:
    """Compute consecutive cosine similarity sim[t] = cos(X[t], X[t-1]) with NaN at t=0."""
    n = X.shape[0]
    sim = np.full(n, np.nan, dtype=float)
    for t in range(1, n):
        sim[t] = float(cosine_similarity(X[t], X[t - 1])[0, 0])
    return sim


def build_output(df: pd.DataFrame, sim_count: np.ndarray, sim_tfidf: np.ndarray) -> pd.DataFrame:
    """Assemble the output dataframe with levels and log-transforms."""
    eps = float(CONFIG["EPS"])
    log_count = np.where(np.isnan(sim_count), np.nan, np.log(sim_count + eps))
    log_tfidf = np.where(np.isnan(sim_tfidf), np.nan, np.log(sim_tfidf + eps))

    return pd.DataFrame({
        "date": df["date"],
        "url": df["url"] if "url" in df.columns else "",
        "sim_countcos": sim_count,
        "sim_tfidfcos": sim_tfidf,
        "log_sim_countcos": log_count,
        "log_sim_tfidfcos": log_tfidf,
        "text_col": CONFIG["TEXT_COL"],
        "ngram_range": [str(CONFIG["NGRAM_RANGE"])] * len(df),
    })


def save_csv(out: pd.DataFrame, out_path: Path) -> None:
    """Write cosine outputs to CSV."""
    out.to_csv(out_path, index=False, encoding="utf-8")


def save_plots(df: pd.DataFrame, out: pd.DataFrame, plots_dir: Path) -> tuple[Path, Path]:
    """Save time-series and scatter plots for cosine similarities."""
    fig_series = plots_dir / "cosine_series.png"
    plt.figure()
    plt.plot(df["date_dt"], out["sim_countcos"], label="Cosine (Counts)")
    plt.plot(df["date_dt"], out["sim_tfidfcos"], label="Cosine (TF-IDF)")
    plt.legend()
    plt.title("Consecutive cosine similarity over time")
    plt.xlabel("Date")
    plt.ylabel("Similarity")
    plt.tight_layout()
    plt.savefig(fig_series, dpi=200)
    plt.close()

    valid = out[["sim_countcos", "sim_tfidfcos"]].dropna()
    fig_scatter = plots_dir / "cosine_scatter.png"
    plt.figure()
    plt.scatter(valid["sim_countcos"], valid["sim_tfidfcos"], s=10)
    plt.title("TF-IDF cosine vs Count cosine (consecutive)")
    plt.xlabel("Cosine (Counts)")
    plt.ylabel("Cosine (TF-IDF)")
    plt.tight_layout()
    plt.savefig(fig_scatter, dpi=200)
    plt.close()

    return fig_series, fig_scatter


def main() -> None:
    """Execute the cosine comparison pipeline and export CSV + plots."""
    project_root = get_project_root()
    in_path, out_path, plots_dir = resolve_paths(project_root)

    df = load_and_filter(in_path)
    texts = df[CONFIG["TEXT_COL"]].fillna("").astype(str).tolist()
    if len(texts) < 2:
        raise ValueError("Not enough documents to compute similarities (need >=2).")

    kwargs = make_vectorizers()
    X_count = CountVectorizer(**kwargs).fit_transform(texts)
    X_tfidf = TfidfVectorizer(**kwargs).fit_transform(texts)

    sim_count = consecutive_cosine(X_count)
    sim_tfidf = consecutive_cosine(X_tfidf)

    out = build_output(df, sim_count, sim_tfidf)
    save_csv(out, out_path)

    valid = out[["sim_countcos", "sim_tfidfcos"]].dropna()
    corr = valid["sim_countcos"].corr(valid["sim_tfidfcos"])
    diff = valid["sim_tfidfcos"] - valid["sim_countcos"]

    fig_series, fig_scatter = save_plots(df, out, plots_dir)

    print(f"Saved: {out_path}")
    print(f"Saved figure: {fig_series}")
    print(f"Saved figure: {fig_scatter}")
    print(f"n_docs={len(df)} | n_pairs={len(valid)} | corr={corr:.3f}")
    print(f"diff(tfidf-count): mean={diff.mean():.4f} median={diff.median():.4f} std={diff.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
