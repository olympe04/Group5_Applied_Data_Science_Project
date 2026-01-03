# E5_jaccard_learning_speed.py
# Compute learning-speed measures using BASE Jaccard bigrams similarity:
#   (1) 1-lag Jaccard bigrams similarity
#   (2) k-memory centroid similarity where centroid = UNION of previous k bigram sets
#   novelty = 1 - centroid similarity
#
# I/O:
#   Input : data_clean/ecb_statements_preprocessed.csv
#   Output: data_features/ecb_similarity_jaccard_learning.csv
#           outputs/plots/ts_sim_centroid_k_jaccard.png
#
# Notes:
# - Uses ECB_START_DATE / ECB_END_DATE if set, otherwise DATE_MIN / DATE_MAX.

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "INPUT_CSV": "data_clean/ecb_statements_preprocessed.csv",
    "OUTPUT_CSV": "data_features/ecb_similarity_jaccard_learning.csv",

    # We use stems to match the paper’s Jaccard bigrams step most closely.
    "TEXT_COL": "stems_str",  # "stems_str" or "tokens_clean_str"

    "DATE_MIN": "1999-01-01",
    "DATE_MAX": "2025-12-31",

    "K_MEMORY": 6,
    "EPS": 1e-6,

    "PLOTS_DIR": "outputs/plots",
    "PLOT_DPI": 200,
}


def get_project_root() -> Path:
    """
    Robustly find project root even if this script is moved into subfolders like:
      <root>/extension/learning_speed/E5_jaccard_learning_speed.py
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError(
        "Could not locate project root. Expected to find 'data_clean/' and 'outputs/' in a parent directory."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Jaccard-bigrams learning-speed measures (1-lag + centroid-k).")
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD (overrides env/config)")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD (overrides env/config)")
    p.add_argument("--k", type=int, default=None, help="Memory length k (overrides config)")
    return p.parse_args()


def resolve_window(args: argparse.Namespace) -> tuple[str | None, str | None]:
    start = args.start_date or os.getenv("ECB_START_DATE") or CONFIG.get("DATE_MIN")
    end = args.end_date or os.getenv("ECB_END_DATE") or CONFIG.get("DATE_MAX")
    return start, end


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path]:
    in_path = project_root / CONFIG["INPUT_CSV"]
    out_path = project_root / CONFIG["OUTPUT_CSV"]
    plots_dir = project_root / CONFIG["PLOTS_DIR"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return in_path, out_path, plots_dir


def load_and_filter(in_path: Path, start: str | None, end: str | None) -> pd.DataFrame:
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path} (run preprocess first)")

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


def bigram_set(tokens_str: str) -> set[str]:
    """
    Build a set of bigrams (as strings "w1_w2") from a whitespace-tokenized string.
    """
    if not isinstance(tokens_str, str) or tokens_str.strip() == "":
        return set()
    toks = tokens_str.split()
    if len(toks) < 2:
        return set()
    return {f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return np.nan
    u = a | b
    if not u:
        return np.nan
    return len(a & b) / len(u)


def sim_1lag(sets: list[set[str]]) -> np.ndarray:
    n = len(sets)
    sim = np.full(n, np.nan, dtype=float)
    for t in range(1, n):
        sim[t] = jaccard(sets[t], sets[t - 1])
    return sim


def sim_centroid_union_k(sets: list[set[str]], k: int) -> np.ndarray:
    """
    centroid-k = UNION of previous k sets:
      sim_centroid_k[t] = Jaccard( S[t], UNION(S[t-k:t]) ) for t>=k else NaN.
    """
    n = len(sets)
    sim = np.full(n, np.nan, dtype=float)
    if k <= 0:
        return sim

    for t in range(k, n):
        centroid = set()
        for s in sets[t - k : t]:
            centroid |= s
        sim[t] = jaccard(sets[t], centroid)
    return sim


def build_output(df: pd.DataFrame, sim1: np.ndarray, simk: np.ndarray, k: int) -> pd.DataFrame:
    eps = float(CONFIG["EPS"])
    log_sim1 = np.where(np.isnan(sim1), np.nan, np.log(sim1 + eps))
    log_simk = np.where(np.isnan(simk), np.nan, np.log(simk + eps))
    novelty = np.where(np.isnan(simk), np.nan, 1.0 - simk)

    return pd.DataFrame(
        {
            "date": df["date"],
            "url": df["url"] if "url" in df.columns else "",
            "text_col": CONFIG["TEXT_COL"],
            "k_memory": int(k),

            "sim_jaccard_1lag": sim1,
            "log_sim_jaccard_1lag": log_sim1,

            "sim_centroid_k": simk,
            "log_sim_centroid_k": log_simk,

            "novelty_centroid_k": novelty,
        }
    )


def save_plot(out: pd.DataFrame, plots_dir: Path, k: int) -> Path:
    plot_df = out.copy()
    plot_df["date_dt"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date_dt"]).sort_values("date_dt")

    fig = plt.figure(figsize=(11, 4.5))
    plt.plot(plot_df["date_dt"], pd.to_numeric(plot_df["sim_centroid_k"], errors="coerce"))
    plt.title(f"Jaccard bigrams centroid similarity (k={k})")
    plt.xlabel("Date")
    plt.ylabel("sim_centroid_k")
    plt.tight_layout()

    p = plots_dir / "ts_sim_centroid_k_jaccard.png"
    fig.savefig(p, dpi=int(CONFIG["PLOT_DPI"]))
    plt.close(fig)
    return p


def main() -> None:
    args = parse_args()
    start, end = resolve_window(args)
    k = int(args.k) if args.k is not None else int(CONFIG["K_MEMORY"])

    project_root = get_project_root()
    in_path, out_path, plots_dir = resolve_paths(project_root)

    df = load_and_filter(in_path, start, end)

    texts = df[CONFIG["TEXT_COL"]].fillna("").astype(str).tolist()
    if len(texts) < 2:
        raise ValueError(f"Not enough docs (n={len(texts)}) to compute similarities.")

    sets = [bigram_set(s) for s in texts]

    sim1 = sim_1lag(sets)
    simk = sim_centroid_union_k(sets, k=k)

    out = build_output(df, sim1, simk, k=k)
    out.to_csv(out_path, index=False, encoding="utf-8")
    p_plot = save_plot(out, plots_dir, k=k)

    print(f"Saved: {out_path}")
    print(f"Saved plot: {p_plot}")

    v1 = out["sim_jaccard_1lag"].dropna()
    vk = out["sim_centroid_k"].dropna()
    print(f"n_docs={len(out)} | n_1lag={len(v1)} | n_centroid_k={len(vk)}")
    if len(vk):
        print(f"sim_centroid_k (k={k}) min={vk.min():.4f} mean={vk.mean():.4f} max={vk.max():.4f}")


if __name__ == "__main__":
    main()
