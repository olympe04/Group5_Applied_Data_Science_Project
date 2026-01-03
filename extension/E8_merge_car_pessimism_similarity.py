# E8_merge_car_pessimism_similarity.py
# Merge event-study CAR/pessimism with TF-IDF cosine similarity into one analysis dataset.
# I/O:
#   Inputs : data_clean/ecb_pessimism_with_car.csv, data_features/ecb_similarity_tfidf.csv
#   Outputs: data_clean/ecb_analysis_dataset.csv
# Notes:
#   Dates are aligned at the event date, the TF-IDF series is merged (left-join), and a z-score version is added.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CONFIG = {
    "INPUT_CAR_PESS": "data_clean/ecb_pessimism_with_car.csv",
    "INPUT_TFIDF_SIM": "data_features/ecb_similarity_tfidf.csv",
    "OUTPUT_MERGED": "data_clean/ecb_analysis_dataset.csv",
    "EPS": 1e-6,
    "DATE_MIN": "1999-01-01",   # set None to disable
    "DATE_MAX": "2013-12-31",   # set None to disable
}


def get_project_root() -> Path:
    """Return repository root (script is in extension/)."""
    return Path(__file__).resolve().parent.parent


def read_csv_checked(path: Path) -> pd.DataFrame:
    """Read a CSV and raise a clean error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def normalize_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Parse a date column and store it as YYYY-MM-DD strings (dropping unparseable rows)."""
    out = df.copy()
    dt = pd.to_datetime(out[col], errors="coerce")
    out = out.loc[dt.notna()].copy()
    out[col] = dt.loc[dt.notna()].dt.strftime("%Y-%m-%d")
    return out


def filter_window(df: pd.DataFrame, dmin: str | None, dmax: str | None) -> pd.DataFrame:
    """Filter dataframe to [dmin, dmax] on its 'date' column (inclusive)."""
    if not dmin and not dmax:
        return df
    dt = pd.to_datetime(df["date"], errors="coerce")
    m = dt.notna()
    if dmin:
        m &= dt >= pd.to_datetime(dmin)
    if dmax:
        m &= dt <= pd.to_datetime(dmax)
    return df.loc[m].copy()


def ensure_log_sim(sim: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Ensure 'log_sim_tfidf' exists in similarity df (computed from sim_tfidf if needed)."""
    sim2 = sim.copy()
    if "sim_tfidf" not in sim2.columns:
        raise ValueError("TF-IDF file must contain 'sim_tfidf' to merge.")
    if "log_sim_tfidf" not in sim2.columns:
        sim2["log_sim_tfidf"] = np.log(pd.to_numeric(sim2["sim_tfidf"], errors="coerce") + eps)
    return sim2


def add_zscore(merged: pd.DataFrame, col: str = "sim_tfidf") -> pd.DataFrame:
    """Add z-score of a column as 'z_<col>' if standard deviation is non-zero."""
    out = merged.copy()
    s = pd.to_numeric(out[col], errors="coerce")
    sd = s.std(ddof=1)
    out[f"z_{col}"] = (s - s.mean()) / sd if sd and not np.isnan(sd) and sd > 0 else np.nan
    return out


def merge_datasets(car: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """Left-merge CAR/pessimism with similarity on the 'date' column."""
    sim_keep = sim[["date", "sim_tfidf", "log_sim_tfidf"]].copy()
    return car.merge(sim_keep, on="date", how="left")


def main() -> None:
    """Load inputs, normalize dates, merge, optionally filter the window, add z-score, and save CSV."""
    project_root = get_project_root()

    p_car = project_root / CONFIG["INPUT_CAR_PESS"]
    p_sim = project_root / CONFIG["INPUT_TFIDF_SIM"]
    p_out = project_root / CONFIG["OUTPUT_MERGED"]
    p_out.parent.mkdir(parents=True, exist_ok=True)

    car = read_csv_checked(p_car)
    sim = read_csv_checked(p_sim)

    if "date" not in car.columns:
        raise ValueError(f"Missing 'date' in {p_car}")
    if "date" not in sim.columns:
        raise ValueError(f"Missing 'date' in {p_sim}")

    car = normalize_date(car, "date")
    sim = ensure_log_sim(normalize_date(sim, "date"), float(CONFIG["EPS"]))

    merged = merge_datasets(car, sim)
    merged = filter_window(merged, CONFIG["DATE_MIN"], CONFIG["DATE_MAX"])
    merged = add_zscore(merged, "sim_tfidf")

    print(f"Rows: {len(merged)}")
    print(f"Missing sim_tfidf: {int(merged['sim_tfidf'].isna().sum())}")

    merged.to_csv(p_out, index=False, encoding="utf-8")
    print(f"Saved: {p_out}")


if __name__ == "__main__":
    main()