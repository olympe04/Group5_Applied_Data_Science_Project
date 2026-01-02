# 7b_prepare_controls_month_end.py
# Build monthly control variables aligned to end-of-month (EOM) timing for event-study regressions.
# I/O:
#   Inputs: data_raw/AMECO-AVGDGP-EA12.csv, data_raw/HICP_data_base100_2005.csv, data_raw/MRO.csv.
#   Outputs: data_clean/controls_month_end.csv with monthly controls (HICP, inflation, annual output gap, EOM MRO level, and EOM delta).

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_one_col_semicolon(path: Path, col_names):
    """Parse files that may store semicolon-separated fields inside a single CSV column."""
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        col = df.columns[0]
        tmp = df[col].astype(str).str.split(";", n=len(col_names) - 1, expand=True)
        tmp.columns = col_names
        return tmp
    return df


def load_output_gap_annual(path: Path) -> pd.DataFrame:
    """Load annual output gap by year from an AMECO-style CSV and return columns (year, output_gap)."""
    tmp = parse_one_col_semicolon(path, ["period", "output_gap"])
    tmp["year"] = pd.to_numeric(tmp["period"], errors="coerce")
    tmp["output_gap"] = pd.to_numeric(tmp["output_gap"], errors="coerce")
    out = tmp.dropna(subset=["year", "output_gap"]).copy()
    out["year"] = out["year"].astype(int)
    out = out[["year", "output_gap"]].drop_duplicates("year").sort_values("year")
    return out


def load_hicp_monthly(path: Path) -> pd.DataFrame:
    """Load monthly HICP index (YYYY-MM) and compute YoY inflation, returning (date_m, hicp, inflation)."""
    tmp = parse_one_col_semicolon(path, ["TIME_PERIOD", "OBS_VALUE"])
    tmp["date_m"] = pd.to_datetime(tmp["TIME_PERIOD"], format="%Y-%m", errors="coerce")
    tmp["hicp"] = pd.to_numeric(tmp["OBS_VALUE"], errors="coerce")
    out = tmp.dropna(subset=["date_m", "hicp"]).copy()
    out["date_m"] = out["date_m"].dt.to_period("M").dt.to_timestamp()
    out = out[["date_m", "hicp"]].drop_duplicates("date_m").sort_values("date_m")
    out["inflation"] = 100.0 * (out["hicp"] / out["hicp"].shift(12) - 1.0)
    return out


def load_mro_daily(path: Path) -> pd.DataFrame:
    """Load daily MRO level from CSV and return columns (date, mro_level) with numeric types."""
    mro = pd.read_csv(path)
    rate_col = next(c for c in mro.columns if "Main refinancing operations" in c)
    mro["date"] = pd.to_datetime(mro["DATE"], errors="coerce")
    mro["mro_level"] = pd.to_numeric(mro[rate_col], errors="coerce")
    mro = mro.dropna(subset=["date", "mro_level"]).sort_values("date").copy()
    return mro[["date", "mro_level"]]


def main() -> None:
    """Assemble monthly controls by merging HICP, annual output gap, and end-of-month MRO levels into one CSV."""
    scripts_dir = Path(__file__).resolve().parent   # .../replication
    project_root = scripts_dir.parent               # .../ (repo root)

    data_raw = project_root / "data_raw"
    data_clean = project_root / "data_clean"
    data_clean.mkdir(parents=True, exist_ok=True)

    og_path = data_raw / "AMECO-AVGDGP-EA12.csv"
    hicp_path = data_raw / "HICP_data_base100_2005.csv"
    mro_path = data_raw / "MRO.csv"

    if not og_path.exists():
        raise FileNotFoundError(f"Missing: {og_path}")
    if not hicp_path.exists():
        raise FileNotFoundError(f"Missing: {hicp_path}")
    if not mro_path.exists():
        raise FileNotFoundError(f"Missing: {mro_path}")

    output_gap = load_output_gap_annual(og_path)
    hicp = load_hicp_monthly(hicp_path)
    mro = load_mro_daily(mro_path)

    mro_m = mro.copy()
    mro_m["date_m"] = mro_m["date"].dt.to_period("M").dt.to_timestamp()
    mro_eom = mro_m.sort_values("date").groupby("date_m", as_index=False).last()
    mro_eom = mro_eom.rename(columns={"mro_level": "mro_level_eom"})[["date_m", "mro_level_eom"]]
    mro_eom = mro_eom.sort_values("date_m").copy()
    mro_eom["delta_mro_eom"] = mro_eom["mro_level_eom"].diff()

    controls = hicp.merge(mro_eom, on="date_m", how="left")
    controls["year"] = controls["date_m"].dt.year
    controls["month"] = controls["date_m"].dt.month
    controls = controls.merge(output_gap, on="year", how="left")
    controls = controls[controls["date_m"] >= "1999-01-01"].copy()

    out_path = data_clean / "controls_month_end.csv"
    controls.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()