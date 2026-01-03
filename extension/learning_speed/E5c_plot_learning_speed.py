# E5c_plot_learning_speed.py
# Plot learning-speed measures (centroid-k similarity and novelty) + comparison vs 1-lag similarity,
# but using the BASE Jaccard bigrams similarity (ecb_similarity_jaccard_bigrams.csv) instead of TF-IDF.
#
# I/O:
#   Input : data_features/ecb_similarity_jaccard_learning.csv   (produced by your learning-speed step)
#           data_clean/ecb_similarity_jaccard_bigrams.csv       (base 1-lag Jaccard series)
#   Output: outputs/plots/ts_sim_centroid_k_jaccard.png
#           outputs/plots/ts_novelty_centroid_k_jaccard.png
#           outputs/plots/ts_compare_1lag_vs_centroidk_jaccard.png

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CFG = {
    "START_DEFAULT": "1999-01-01",
    "END_DEFAULT": "2013-12-31",

    # learning-speed outputs (centroid-k + novelty) that YOU compute in learning_speed step
    "IN_LEARNING_FILE": "data_features/ecb_similarity_jaccard_learning.csv",

    # base similarity (1-lag) to compare against
    "IN_BASE_FILE": "data_clean/ecb_similarity_jaccard_bigrams.csv",

    "OUT_DIR": "outputs/plots",
    "DPI": 200,
}


def get_project_root() -> Path:
    """
    Robustly find project root even if this script is in:
      <root>/extension/learning_speed/E5c_plot_learning_speed.py
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError(
        "Could not locate project root. Expected to find 'data_clean/' and 'outputs/' in a parent directory."
    )


def window() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    s = os.getenv("ECB_START_DATE", CFG["START_DEFAULT"])
    e = os.getenv("ECB_END_DATE", CFG["END_DEFAULT"])
    return pd.Timestamp(s), pd.Timestamp(e), s, e


def main() -> None:
    project_root = get_project_root()
    out_dir = project_root / CFG["OUT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)

    sdt, edt, s, e = window()

    p_learn = project_root / CFG["IN_LEARNING_FILE"]
    p_base = project_root / CFG["IN_BASE_FILE"]

    if not p_learn.exists():
        raise FileNotFoundError(f"Missing: {p_learn} (run your learning-speed step first)")
    if not p_base.exists():
        raise FileNotFoundError(f"Missing: {p_base} (run base similarity step 5 first)")

    # learning-speed dataset (centroid-k + novelty)
    df = pd.read_csv(p_learn)
    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' in {p_learn}")
    if "sim_centroid_k" not in df.columns:
        raise ValueError(f"Missing 'sim_centroid_k' in {p_learn}")
    if "novelty_centroid_k" not in df.columns:
        raise ValueError(f"Missing 'novelty_centroid_k' in {p_learn}")

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values("date_dt")
    df = df[(df["date_dt"] >= sdt) & (df["date_dt"] <= edt)].copy()
    if len(df) == 0:
        raise ValueError(f"No rows in learning-speed plot window {s}..{e} from {p_learn}")

    # --- base 1-lag Jaccard series ---
    base = pd.read_csv(p_base)
    if "date" not in base.columns:
        raise ValueError(f"Missing 'date' in {p_base}")
    if "sim_jaccard" not in base.columns:
        raise ValueError(f"Missing 'sim_jaccard' in {p_base} (expected base 1-lag series)")

    base["date_dt"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["date_dt"]).sort_values("date_dt")
    base = base[(base["date_dt"] >= sdt) & (base["date_dt"] <= edt)].copy()

    # merge base similarity onto learning-speed dates (left join)
    df = df.merge(base[["date_dt", "sim_jaccard"]], on="date_dt", how="left")

    k = int(df["k_memory"].dropna().iloc[0]) if "k_memory" in df.columns and df["k_memory"].notna().any() else None
    title_k = f"(k={k})" if k is not None else ""

    # Plot 1: centroid-k similarity
    fig = plt.figure(figsize=(11, 4.5))
    plt.plot(df["date_dt"], pd.to_numeric(df["sim_centroid_k"], errors="coerce"))
    plt.title(f"Jaccard centroid similarity {title_k}")
    plt.xlabel("Date")
    plt.ylabel("sim_centroid_k")
    plt.tight_layout()
    f1 = out_dir / "ts_sim_centroid_k_jaccard.png"
    fig.savefig(f1, dpi=int(CFG["DPI"]))
    plt.close(fig)

    # Plot 2: novelty = 1 - centroid similarity
    fig = plt.figure(figsize=(11, 4.5))
    plt.plot(df["date_dt"], pd.to_numeric(df["novelty_centroid_k"], errors="coerce"))
    plt.title(f"Novelty (1 - centroid similarity) {title_k}")
    plt.xlabel("Date")
    plt.ylabel("novelty_centroid_k")
    plt.tight_layout()
    f2 = out_dir / "ts_novelty_centroid_k_jaccard.png"
    fig.savefig(f2, dpi=int(CFG["DPI"]))
    plt.close(fig)

    # Plot 3: compare 1-lag (base Jaccard) vs centroid-k
    fig = plt.figure(figsize=(11, 4.5))
    plt.plot(df["date_dt"], pd.to_numeric(df["sim_jaccard"], errors="coerce"))
    plt.plot(df["date_dt"], pd.to_numeric(df["sim_centroid_k"], errors="coerce"))
    plt.title(f"Similarity comparison: 1-lag Jaccard vs centroid {title_k}")
    plt.xlabel("Date")
    plt.ylabel("similarity")
    plt.tight_layout()
    f3 = out_dir / "ts_compare_1lag_jaccard_vs_centroidk.png"
    fig.savefig(f3, dpi=int(CFG["DPI"]))
    plt.close(fig)

    print(f"Saved: {f1}")
    print(f"Saved: {f2}")
    print(f"Saved: {f3}")
    print(f"Window: {s}->{e} | n={len(df)} | base_missing={int(df['sim_jaccard'].isna().sum())}")


if __name__ == "__main__":
    main()
