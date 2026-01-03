# 9_summary_table2.py
# Create a Table 2-style summary statistics table for CAR, similarity, pessimism, and macro controls.
# I/O:
#   Inputs : data_clean/ecb_pessimism_with_car.csv, data_clean/ecb_similarity_jaccard_bigrams.csv, data_clean/controls_month_end.csv
#   Outputs: outputs/tables/table2_summary_stats.csv
# Notes:
#   The script merges event-level CAR/pessimism with similarity and monthly controls, filters to the env window
#   (ECB_START_DATE/ECB_END_DATE), and exports descriptive statistics for the main variables.

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


CONFIG = {
    "DEFAULT_START_DATE": "1999-01-01",
    "DEFAULT_END_DATE": "2013-12-31",
}


def get_project_root() -> Path:
    """Return repository root (script is in replication/)."""
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir.parent


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path, Path]:
    """Return (car_path, sim_path, ctl_path, out_tables_dir)."""
    dc = project_root / "data_clean"
    out_tables = project_root / "outputs" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    return (
        dc / "ecb_pessimism_with_car.csv",
        dc / "ecb_similarity_jaccard_bigrams.csv",
        dc / "controls_month_end.csv",
        out_tables,
    )


def get_window_from_env() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """Return (start_dt, end_dt, start_str, end_str) from env (or defaults)."""
    start_str = os.getenv("ECB_START_DATE", CONFIG["DEFAULT_START_DATE"])
    end_str = os.getenv("ECB_END_DATE", CONFIG["DEFAULT_END_DATE"])
    start_dt, end_dt = pd.Timestamp(start_str), pd.Timestamp(end_str)
    if end_dt < start_dt:
        raise ValueError(f"Invalid window: end < start ({start_str} .. {end_str})")
    return start_dt, end_dt, start_str, end_str


def load_inputs(car_path: Path, sim_path: Path, ctl_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CAR/pessimism, similarity, and controls datasets with parsed date columns."""
    for p, msg in [
        (car_path, "Missing CAR+sentiment file (run step 7 first)"),
        (sim_path, "Missing similarity file (run step 5 first)"),
        (ctl_path, "Missing controls file (run step 7b first)"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{msg}: {p}")

    car = pd.read_csv(car_path, parse_dates=["date"])
    sim = pd.read_csv(sim_path, parse_dates=["date"])
    ctl = pd.read_csv(ctl_path, parse_dates=["date_m"])
    return car, sim, ctl


def build_event_df(
    car: pd.DataFrame, sim: pd.DataFrame, ctl: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> pd.DataFrame:
    """Merge event-level datasets, attach monthly controls, and filter to the env window (sim_jaccard>0)."""
    df = (
        car.merge(sim[["date", "sim_jaccard"]], on="date", how="left")
           .assign(date_m=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
           .merge(ctl[["date_m", "output_gap", "inflation", "delta_mro_eom"]], on="date_m", how="left")
           .dropna(subset=["date", "sim_jaccard"])
           .loc[lambda d: (d["date"] >= start_dt) & (d["date"] <= end_dt)]
           .query("sim_jaccard > 0")
           .copy()
    )
    if len(df) == 0:
        raise ValueError("No events available for summary table after filtering (sim_jaccard>0) in the window.")
    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Table 2-style descriptive statistics for main variables and return a formatted table."""
    cols = [
        ("CAR", "CAR_pct"),
        ("|CAR|", "absCAR_pct"),
        ("Pessimism", "pessimism_lm_pct"),
        ("Similarity", "sim_jaccard"),
        ("Output gap", "output_gap"),
        ("Inflation", "inflation"),
        ("Delta MRO", "delta_mro_eom"),
    ]

    stats = pd.DataFrame(
        {name: df[col].astype(float).describe(percentiles=[0.25, 0.5, 0.75]) for name, col in cols}
    ).T.rename(
        columns={
            "mean": "Mean",
            "std": "Std. dev.",
            "min": "Min.",
            "25%": "Quartile 1",
            "50%": "Median",
            "75%": "Quartile 3",
            "max": "Max.",
        }
    )[["Mean", "Std. dev.", "Min.", "Quartile 1", "Median", "Quartile 3", "Max."]].round(2)

    return stats


def save_table(stats: pd.DataFrame, out_tables: Path) -> Path:
    """Save the Table 2 summary stats CSV to outputs/tables and return the written path."""
    out_path = out_tables / "table2_summary_stats.csv"
    stats.to_csv(out_path, encoding="utf-8")
    return out_path


def main() -> None:
    """Execute Table 2 summary stats build and export the formatted table."""
    project_root = get_project_root()
    car_path, sim_path, ctl_path, out_tables = resolve_paths(project_root)
    start_dt, end_dt, start_str, end_str = get_window_from_env()

    car, sim, ctl = load_inputs(car_path, sim_path, ctl_path)
    df = build_event_df(car, sim, ctl, start_dt, end_dt)
    stats = compute_summary_stats(df)
    out_path = save_table(stats, out_tables)

    print(f"Window (env): {start_str} -> {end_str} | N events used = {len(df)}")
    print(f"Saved: {out_path}")
    print(stats.to_string())


if __name__ == "__main__":
    main()