# 05_similarity_jaccard_bigrams.py
# Compute Jaccard similarity between consecutive ECB texts using stem bigrams.
# I/O:
#   Inputs : data_clean/ecb_statements_preprocessed.csv (columns: date, stems_str).
#   Outputs: data_clean/ecb_similarity_jaccard_bigrams.csv (+ optional window plot in outputs/plots).
# Notes:
#   The script deduplicates by date (keeps the longest text), computes bigram Jaccard similarity vs the previous date,
#   saves the full series, and optionally plots the similarity over the env window (ECB_START_DATE/ECB_END_DATE).

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONFIG = {
    "INPUT_CSV": "data_clean/ecb_statements_preprocessed.csv",
    "OUTPUT_CSV": "data_clean/ecb_similarity_jaccard_bigrams.csv",
    "OUTPUT_DIR": "outputs/plots",
    "DEFAULT_START_DATE": "1999-01-01",
    "DEFAULT_END_DATE": "2013-12-31",
    "PLOT": True,
    "SHOW_PLOT": True,
    "PLOT_DPI": 200,
}


def get_project_root() -> Path:
    """Return repository root (script is in replication/)."""
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir.parent


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path]:
    """Return (input_csv_path, output_csv_path, plots_dir_path)."""
    in_path = project_root / CONFIG["INPUT_CSV"]
    out_csv = project_root / CONFIG["OUTPUT_CSV"]
    out_dir = project_root / CONFIG["OUTPUT_DIR"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return in_path, out_csv, out_dir


def get_window_from_env() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """Return (start_dt, end_dt, start_str, end_str) from env (or defaults)."""
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


def load_input(in_path: Path) -> pd.DataFrame:
    """Load preprocessed statements and enforce date/stems formatting."""
    if not in_path.exists():
        raise FileNotFoundError(f"Input introuvable: {in_path} (run preprocessing step first)")

    df = pd.read_csv(in_path)
    df["date"] = df["date"].fillna("").astype(str).str.slice(0, 10)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["stems_str"] = df["stems_str"].fillna("").astype(str)

    df = df[(df["date_dt"].notna()) & (df["stems_str"].str.len() > 0)].copy()
    return df


def dedupe_by_date_keep_longest(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate rows by date, keeping the text with the most stems (or the longest stem string)."""
    d = df.copy()
    d["len_for_dedupe"] = (
        pd.to_numeric(d["n_stems"], errors="coerce").fillna(0).astype(int)
        if "n_stems" in d.columns
        else d["stems_str"].str.split().apply(len)
    )
    return (
        d.sort_values(["date_dt", "len_for_dedupe"], ascending=[True, False])
         .drop_duplicates(subset=["date_dt"], keep="first")
         .sort_values("date_dt")
         .reset_index(drop=True)
    )


def compute_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute consecutive Jaccard similarities over stem bigrams and return an output-ready dataframe."""
    d = df.copy()
    d["bigrams"] = d["stems_str"].apply(build_bigrams_from_stems)
    d["n_bigrams"] = d["bigrams"].apply(len)

    d["prev_date_dt"] = d["date_dt"].shift(1)
    d["n_bigrams_prev"] = d["n_bigrams"].shift(1)

    sims = [float("nan")] + [jaccard(d.at[i, "bigrams"], d.at[i - 1, "bigrams"]) for i in range(1, len(d))]
    d["sim_jaccard"] = sims

    out = d[["date_dt", "prev_date_dt", "sim_jaccard", "n_bigrams", "n_bigrams_prev"]].copy()
    out = out.rename(columns={"date_dt": "date", "prev_date_dt": "prev_date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["prev_date"] = pd.to_datetime(out["prev_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def write_output(out_df: pd.DataFrame, out_csv: Path) -> None:
    """Write the full-sample similarity series to CSV."""
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved FULL sample: {out_csv} | rows={len(out_df)}")


def plot_window(df_full: pd.DataFrame, plots_dir: Path) -> None:
    """Plot similarity over the env window and save the figure to outputs/plots."""
    start_dt, end_dt, start_str, end_str = get_window_from_env()
    w = df_full[(df_full["date_dt"] >= start_dt) & (df_full["date_dt"] <= end_dt) & (df_full["sim_jaccard"].notna())]
    print(f"Plot observations: {len(w)}")

    plt.figure()
    plt.plot(w["date_dt"], w["sim_jaccard"], linewidth=1)
    plt.title(f"ECB text similarity (Jaccard bigrams), {start_str}–{end_str}")
    plt.xlabel("Date")
    plt.ylabel("Similarity")
    plt.tight_layout()

    fig_path = plots_dir / f"similarity_jaccard_bigrams_{start_str.replace('-', '')}_{end_str.replace('-', '')}.png"
    plt.savefig(fig_path, dpi=int(CONFIG["PLOT_DPI"]))
    plt.show() if CONFIG["SHOW_PLOT"] else plt.close()
    print(f"Saved plot (window-only): {fig_path}")


def main() -> None:
    """Execute similarity computation and optional plotting."""
    project_root = get_project_root()
    in_path, out_csv, plots_dir = resolve_paths(project_root)

    df = load_input(in_path)
    df = dedupe_by_date_keep_longest(df)

    out_df = compute_similarity(df)
    write_output(out_df, out_csv)

    if CONFIG["PLOT"]:
        df_plot = df.copy()
        df_plot["sim_jaccard"] = pd.to_numeric([None] + out_df["sim_jaccard"].iloc[1:].tolist(), errors="coerce")
        df_plot["sim_jaccard"] = pd.to_numeric(df_plot["sim_jaccard"], errors="coerce")
        plot_window(df_plot, plots_dir)


if __name__ == "__main__":
    main()