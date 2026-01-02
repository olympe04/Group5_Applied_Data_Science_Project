# 05_similarity_jaccard_bigrams.py
# Compute Jaccard similarity between consecutive ECB texts using stem bigrams.
# I/O:
#   Inputs: data_clean/ecb_statements_preprocessed.csv (columns: date, stems_str).
#   Outputs: data_clean/ecb_similarity_jaccard_bigrams.csv (+ optional window plot in outputs/).

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONFIG = {
    "INPUT_CSV": "data_clean/ecb_statements_preprocessed.csv",
    "OUTPUT_CSV": "data_clean/ecb_similarity_jaccard_bigrams.csv",
    "OUTPUT_DIR": "outputs",
    "DEFAULT_START_DATE": "1999-01-01",
    "DEFAULT_END_DATE": "2013-12-31",
    "PLOT": True,
    "SHOW_PLOT": True,
    "PLOT_DPI": 200,
}


def get_window_from_env() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """Read (start, end) plot window from env (or defaults) and return parsed timestamps plus the original strings."""
    start_str = os.getenv("ECB_START_DATE", CONFIG["DEFAULT_START_DATE"])
    end_str = os.getenv("ECB_END_DATE", CONFIG["DEFAULT_END_DATE"])
    start_dt = pd.Timestamp(start_str)
    end_dt = pd.Timestamp(end_str)
    if end_dt < start_dt:
        raise ValueError(f"Invalid window: end < start ({start_str} .. {end_str})")
    return start_dt, end_dt, start_str, end_str


def jaccard(a: set, b: set) -> float:
    """Return Jaccard(a, b) = |a∩b|/|a∪b| (NaN if both sets are empty)."""
    a = a or set()
    b = b or set()
    return float("nan") if not (a or b) else (len(a & b) / len(a | b))


def build_bigrams_from_stems(stems_str: str) -> set:
    """Build consecutive token bigrams from a space-separated stem string."""
    toks = str(stems_str).split()
    return set(zip(toks[:-1], toks[1:])) if len(toks) >= 2 else set()


def main() -> None:
    """Compute consecutive-document bigram Jaccard similarities, save CSV, and optionally plot a window."""
    scripts_dir = Path(__file__).resolve().parent   # .../replication
    project_root = scripts_dir.parent               # .../ (repo root)

    in_path = project_root / CONFIG["INPUT_CSV"]
    out_csv = project_root / CONFIG["OUTPUT_CSV"]
    out_dir = project_root / CONFIG["OUTPUT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input introuvable: {in_path} (run preprocessing step first)")

    df = pd.read_csv(in_path)

    df["date"] = df["date"].fillna("").astype(str).str.slice(0, 10)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["stems_str"] = df["stems_str"].fillna("").astype(str)

    df = df[(df["date_dt"].notna()) & (df["stems_str"].str.len() > 0)].copy()

    df["len_for_dedupe"] = (
        pd.to_numeric(df["n_stems"], errors="coerce").fillna(0).astype(int)
        if "n_stems" in df.columns
        else df["stems_str"].str.split().apply(len)
    )

    df = (
        df.sort_values(["date_dt", "len_for_dedupe"], ascending=[True, False])
          .drop_duplicates(subset=["date_dt"], keep="first")
          .sort_values("date_dt")
          .reset_index(drop=True)
    )

    df["bigrams"] = df["stems_str"].apply(build_bigrams_from_stems)
    df["n_bigrams"] = df["bigrams"].apply(len)

    df["prev_date_dt"] = df["date_dt"].shift(1)
    df["n_bigrams_prev"] = df["n_bigrams"].shift(1)

    sims = [float("nan")]
    for i in range(1, len(df)):
        sims.append(jaccard(df.at[i, "bigrams"], df.at[i - 1, "bigrams"]))
    df["sim_jaccard"] = sims

    out_df = df[["date_dt", "prev_date_dt", "sim_jaccard", "n_bigrams", "n_bigrams_prev"]].copy()
    out_df = out_df.rename(columns={"date_dt": "date", "prev_date_dt": "prev_date"})
    out_df["date"] = out_df["date"].dt.strftime("%Y-%m-%d")
    out_df["prev_date"] = pd.to_datetime(out_df["prev_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved FULL sample: {out_csv} | rows={len(out_df)}")

    if CONFIG["PLOT"]:
        start_dt, end_dt, start_str, end_str = get_window_from_env()
        w = df[(df["date_dt"] >= start_dt) & (df["date_dt"] <= end_dt) & (df["sim_jaccard"].notna())]
        print(f"Plot observations: {len(w)}")

        plt.figure()
        plt.plot(w["date_dt"], w["sim_jaccard"], linewidth=1)
        plt.title(f"ECB text similarity (Jaccard bigrams), {start_str}–{end_str}")
        plt.xlabel("Date")
        plt.ylabel("Similarity")
        plt.tight_layout()

        fig_path = out_dir / f"similarity_jaccard_bigrams_{start_str.replace('-', '')}_{end_str.replace('-', '')}.png"
        plt.savefig(fig_path, dpi=int(CONFIG["PLOT_DPI"]))
        plt.show() if CONFIG["SHOW_PLOT"] else plt.close()
        print(f"Saved plot (window-only): {fig_path}")


if __name__ == "__main__":
    main()
