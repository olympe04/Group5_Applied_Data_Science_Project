# 7b_prepare_controls_month_end.py
# Build monthly control variables aligned to end-of-month (EOM) timing for event-study regressions.
# I/O:
#   Inputs : data_raw/AMECO-AVGDGP-EA12.csv, data_raw/HICP_data_base100_2005.csv, data_raw/MRO.csv
#   Outputs: data_clean/controls_month_end.csv with monthly controls (hicp, inflation, output_gap, mro_level_eom, delta_mro_eom)
# Notes:
#   The script parses the three raw sources, computes YoY inflation from HICP, keeps MRO at month-end,
#   merges annual output gap by year, and exports a single monthly controls table.

from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    """Return repository root (script is in replication/)."""
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir.parent


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path, Path]:
    """Return (raw_dir, clean_dir, og_path, hicp_path, mro_path) paths for the controls build."""
    raw_dir = project_root / "data_raw"
    clean_dir = project_root / "data_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    return (
        raw_dir,
        clean_dir,
        raw_dir / "AMECO-AVGDGP-EA12.csv",
        raw_dir / "HICP_data_base100_2005.csv",
        raw_dir / "MRO.csv",
    )


def parse_one_col_semicolon(path: Path, col_names: list[str]) -> pd.DataFrame:
    """Parse CSVs that may store semicolon-separated fields inside a single column."""
    df = pd.read_csv(path)
    if df.shape[1] != 1:
        return df
    col = df.columns[0]
    tmp = df[col].astype(str).str.split(";", n=len(col_names) - 1, expand=True)
    tmp.columns = col_names
    return tmp


def require_files(paths: list[Path]) -> None:
    """Raise FileNotFoundError if any required input file is missing."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n" + "\n".join(f"  - {p}" for p in missing))


def load_output_gap_annual(path: Path) -> pd.DataFrame:
    """Load annual output gap from AMECO-like CSV and return columns (year, output_gap)."""
    tmp = parse_one_col_semicolon(path, ["period", "output_gap"])
    tmp["year"] = pd.to_numeric(tmp["period"], errors="coerce")
    tmp["output_gap"] = pd.to_numeric(tmp["output_gap"], errors="coerce")
    out = tmp.dropna(subset=["year", "output_gap"]).copy()
    out["year"] = out["year"].astype(int)
    return out[["year", "output_gap"]].drop_duplicates("year").sort_values("year")


def load_hicp_monthly(path: Path) -> pd.DataFrame:
    """Load monthly HICP (YYYY-MM) and compute YoY inflation, returning (date_m, hicp, inflation)."""
    tmp = parse_one_col_semicolon(path, ["TIME_PERIOD", "OBS_VALUE"])
    tmp["date_m"] = pd.to_datetime(tmp["TIME_PERIOD"], format="%Y-%m", errors="coerce")
    tmp["hicp"] = pd.to_numeric(tmp["OBS_VALUE"], errors="coerce")
    out = tmp.dropna(subset=["date_m", "hicp"]).copy()
    out["date_m"] = out["date_m"].dt.to_period("M").dt.to_timestamp()
    out = out[["date_m", "hicp"]].drop_duplicates("date_m").sort_values("date_m")
    out["inflation"] = 100.0 * (out["hicp"] / out["hicp"].shift(12) - 1.0)
    return out


def load_mro_daily(path: Path) -> pd.DataFrame:
    """Load daily MRO from CSV and return columns (date, mro_level)."""
    mro = pd.read_csv(path)
    rate_col = next(c for c in mro.columns if "Main refinancing operations" in c)
    mro["date"] = pd.to_datetime(mro["DATE"], errors="coerce")
    mro["mro_level"] = pd.to_numeric(mro[rate_col], errors="coerce")
    mro = mro.dropna(subset=["date", "mro_level"]).sort_values("date")
    return mro[["date", "mro_level"]]


def build_mro_eom(mro_daily: pd.DataFrame) -> pd.DataFrame:
    """Convert daily MRO to end-of-month level and month-to-month delta."""
    m = mro_daily.copy()
    m["date_m"] = m["date"].dt.to_period("M").dt.to_timestamp()
    eom = m.sort_values("date").groupby("date_m", as_index=False).last()
    eom = eom.rename(columns={"mro_level": "mro_level_eom"})[["date_m", "mro_level_eom"]].sort_values("date_m")
    eom["delta_mro_eom"] = eom["mro_level_eom"].diff()
    return eom


def merge_controls(hicp: pd.DataFrame, mro_eom: pd.DataFrame, output_gap: pd.DataFrame) -> pd.DataFrame:
    """Merge HICP, MRO (EOM), and annual output gap into a monthly controls dataframe."""
    controls = hicp.merge(mro_eom, on="date_m", how="left")
    controls["year"] = controls["date_m"].dt.year
    controls["month"] = controls["date_m"].dt.month
    controls = controls.merge(output_gap, on="year", how="left")
    return controls.loc[controls["date_m"] >= "1999-01-01"].copy()


def write_output(df: pd.DataFrame, clean_dir: Path) -> Path:
    """Write controls_month_end.csv to data_clean and return the written path."""
    out_path = clean_dir / "controls_month_end.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def main() -> None:
    """Execute controls construction and write data_clean/controls_month_end.csv."""
    project_root = get_project_root()
    raw_dir, clean_dir, og_path, hicp_path, mro_path = resolve_paths(project_root)

    require_files([og_path, hicp_path, mro_path])

    output_gap = load_output_gap_annual(og_path)
    hicp = load_hicp_monthly(hicp_path)
    mro_daily = load_mro_daily(mro_path)

    mro_eom = build_mro_eom(mro_daily)
    controls = merge_controls(hicp, mro_eom, output_gap)

    out_path = write_output(controls, clean_dir)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()