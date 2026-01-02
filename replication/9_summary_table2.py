# 9_summary_table2.py
# Compute Table 2-style summary statistics from CAR, similarity, and controls within an env-defined window.
# I/O:
#   Inputs: data_clean/ecb_pessimism_with_car.csv, data_clean/ecb_similarity_jaccard_bigrams.csv, data_clean/controls_month_end.csv.
#   Outputs: outputs/table2_summary_stats.csv (summary statistics table).

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


CONFIG = {
    "DEFAULT_START_DATE": "1999-01-01",
    "DEFAULT_END_DATE": "2013-12-31",
}


def get_window_from_env() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """Read (start, end) summary window from env (or defaults) and return parsed timestamps plus the original strings."""
    start_str = os.getenv("ECB_START_DATE", CONFIG["DEFAULT_START_DATE"])
    end_str = os.getenv("ECB_END_DATE", CONFIG["DEFAULT_END_DATE"])
    start_dt = pd.Timestamp(start_str)
    end_dt = pd.Timestamp(end_str)
    if end_dt < start_dt:
        raise ValueError(f"Invalid window: end < start ({start_str} .. {end_str})")
    return start_dt, end_dt, start_str, end_str


def main() -> None:
    """Merge event-level datasets and export descriptive statistics for key variables within the configured window."""
    scripts_dir = Path(__file__).resolve().parent   # .../replication
    project_root = scripts_dir.parent               # .../ (repo root)

    dc = project_root / "data_clean"
    out = project_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    start_dt, end_dt, start_str, end_str = get_window_from_env()

    car_path = dc / "ecb_pessimism_with_car.csv"
    sim_path = dc / "ecb_similarity_jaccard_bigrams.csv"
    ctl_path = dc / "controls_month_end.csv"

    if not car_path.exists():
        raise FileNotFoundError(f"Missing: {car_path} (run step 7 first)")
    if not sim_path.exists():
        raise FileNotFoundError(f"Missing: {sim_path} (run step 5 first)")
    if not ctl_path.exists():
        raise FileNotFoundError(f"Missing: {ctl_path} (run step 7b first)")

    car = pd.read_csv(car_path, parse_dates=["date"])
    sim = pd.read_csv(sim_path, parse_dates=["date"])
    ctl = pd.read_csv(ctl_path, parse_dates=["date_m"])

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
        raise ValueError(
            f"No events available for summary table after filtering (sim_jaccard>0) in window {start_str}..{end_str}."
        )

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

    out_path = out / "table2_summary_stats.csv"
    stats.to_csv(out_path, encoding="utf-8")

    print(f"Window (env): {start_str} -> {end_str} | N events used = {len(df)}")
    print(f"Saved: {out_path}")
    print(stats.to_string())


if __name__ == "__main__":
    main()
