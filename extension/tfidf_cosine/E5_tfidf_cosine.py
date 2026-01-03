# E5_tfidf_cosine.py
# Compute consecutive tfidf_cosine similarity using TF-IDF representations.
# I/O:
#   Inputs : data_clean/ecb_statements_preprocessed.csv
#   Outputs: data_features/ecb_similarity_tfidf.csv,
#            outputs/plots/ts_sim_tfidf.png
# Notes:
#   The script builds a TF-IDF matrix on the filtered corpus (date window from CLI/env/config),
#   computes consecutive tfidf_cosine similarity, saves the series, and exports a time-series plot.

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # for headless environments (server/CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CONFIG = {
    "INPUT_CSV": "data_clean/ecb_statements_preprocessed.csv",
    "OUTPUT_CSV": "data_features/ecb_similarity_tfidf.csv",
    "TEXT_COL": "stems_str",          # "stems_str" or "tokens_clean_str"
    "NGRAM_RANGE": (1, 2),
    "MIN_DF": 1,
    "MAX_DF": 0.95,                  # None to disable
    "EPS": 1e-6,
    "DATE_MIN": "1999-01-01",
    "DATE_MAX": "2013-12-31",
    "PLOTS_DIR": "outputs/plots",
}


def get_project_root() -> Path:
    """
    Robustly find project root even if this script is moved into subfolders like:
      <root>/extension/tfidf_cosine/E5_tfidf_cosine.py

    Strategy: walk up parents until we find expected root markers.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError(
        "Could not locate project root. Expected to find 'data_clean/' and 'outputs/' in a parent directory."
    )


def parse_args() -> argparse.Namespace:
    """Parse optional CLI date overrides."""
    p = argparse.ArgumentParser(description="Compute consecutive TF-IDF cosine similarity.")
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD (overrides env/config)")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD (overrides env/config)")
    return p.parse_args()


def resolve_window(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Resolve (start, end) with priority: CLI > env > config."""
    start = args.start_date or os.getenv("ECB_START_DATE") or CONFIG.get("DATE_MIN")
    end = args.end_date or os.getenv("ECB_END_DATE") or CONFIG.get("DATE_MAX")
    return start, end


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path]:
    """Return (input_csv_path, output_csv_path, plots_dir) and create needed directories."""
    in_path = project_root / CONFIG["INPUT_CSV"]
    out_path = project_root / CONFIG["OUTPUT_CSV"]
    plots_dir = project_root / CONFIG["PLOTS_DIR"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return in_path, out_path, plots_dir


def load_and_filter(in_path: Path, start: str | None, end: str | None) -> pd.DataFrame:
    """Load input CSV, parse/sort dates, and apply the date window."""
    if not in_path.exists():
        raise FileNotFoundError(f"Input introuvable: {in_path} (run replication preprocess step first)")

    df = pd.read_csv(in_path)
    if "date" not in df.columns:
        raise ValueError("Input missing 'date' column")
    if CONFIG["TEXT_COL"] not in df.columns:
        raise ValueError(f"Input missing '{CONFIG['TEXT_COL']}' column")

    df = df.copy()
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values("date_dt").reset_index(drop=True)

    if start:
        df = df[df["date_dt"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date_dt"] <= pd.to_datetime(end)]

    df = df.reset_index(drop=True)
    print(f"Date filter: start={start} end={end} | n_docs={len(df)}")
    return df


def build_tfidf_matrix(texts: list[str]):
    """Fit a TF-IDF vectorizer on the corpus and return the sparse matrix."""
    kwargs = dict(
        ngram_range=CONFIG["NGRAM_RANGE"],
        min_df=CONFIG["MIN_DF"],
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    if CONFIG["MAX_DF"] is not None:
        kwargs["max_df"] = CONFIG["MAX_DF"]
    return TfidfVectorizer(**kwargs).fit_transform(texts)


def consecutive_cosine(X) -> np.ndarray:
    """Compute consecutive cosine similarity sim[t] = cos(X[t], X[t-1]) with NaN at t=0."""
    n = X.shape[0]
    sim = np.full(n, np.nan, dtype=float)
    for t in range(1, n):
        sim[t] = float(cosine_similarity(X[t], X[t - 1])[0, 0])
    return sim


def build_output(df: pd.DataFrame, sim: np.ndarray) -> pd.DataFrame:
    """Assemble the output dataframe with levels and log-transform."""
    eps = float(CONFIG["EPS"])
    log_sim = np.where(np.isnan(sim), np.nan, np.log(sim + eps))

    return pd.DataFrame(
        {
            "date": df["date"],
            "url": df["url"] if "url" in df.columns else "",
            "sim_tfidf": sim,
            "log_sim_tfidf": log_sim,
            "text_col": CONFIG["TEXT_COL"],
            "ngram_range": str(CONFIG["NGRAM_RANGE"]),
        }
    )


def save_csv(out: pd.DataFrame, out_path: Path) -> None:
    """Write TF-IDF similarity outputs to CSV."""
    out.to_csv(out_path, index=False, encoding="utf-8")


def save_plot(out: pd.DataFrame, plots_dir: Path) -> Path:
    """Save the time-series plot of sim_tfidf."""
    plot_df = out.copy()
    plot_df["date_dt"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date_dt"]).sort_values("date_dt")

    fig = plt.figure(figsize=(11, 4.5))
    plt.plot(plot_df["date_dt"], pd.to_numeric(plot_df["sim_tfidf"], errors="coerce"))
    plt.title("Consecutive TF-IDF cosine similarity (sim_tfidf)")
    plt.xlabel("Date")
    plt.ylabel("sim_tfidf")
    plt.tight_layout()

    p = plots_dir / "ts_sim_tfidf.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def main() -> None:
    """Execute TF-IDF cosine computation and export CSV + plot."""
    args = parse_args()
    start, end = resolve_window(args)

    project_root = get_project_root()
    in_path, out_path, plots_dir = resolve_paths(project_root)

    df = load_and_filter(in_path, start, end)
    texts = df[CONFIG["TEXT_COL"]].fillna("").astype(str).tolist()
    if len(texts) < 2:
        raise ValueError(
            "Not enough documents to compute consecutive similarities. "
            f"(n_docs={len(texts)} after date filter start={start} end={end})"
        )

    X = build_tfidf_matrix(texts)
    sim = consecutive_cosine(X)

    out = build_output(df, sim)
    save_csv(out, out_path)
    p_plot = save_plot(out, plots_dir)

    valid = out["sim_tfidf"].dropna()
    print(f"Saved: {out_path}")
    print(f"Saved plot: {p_plot}")
    print(f"n_docs={len(df)} | n_sims={len(valid)}")
    if len(valid):
        print(f"sim_tfidf min={valid.min():.4f} mean={valid.mean():.4f} max={valid.max():.4f}")


if __name__ == "__main__":
    main()