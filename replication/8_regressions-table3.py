# 8_regressions-table3.py
# Estimate Table 3-style OLS regressions for log(similarity) within an env-defined observation window.
# I/O:
#   Inputs : data_clean/ecb_similarity_jaccard_bigrams.csv, data_clean/controls_month_end.csv
#   Outputs: outputs/tables/table3_similarity_regressions.csv
# Notes:
#   The script filters the similarity series to the env window, builds time trends, merges monthly controls,
#   runs four OLS specs with HC1 robust SE, and exports a compact CSV table.

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


CONFIG = {
    "DEFAULT_START_DATE": "1999-01-01",
    "DEFAULT_END_DATE": "2013-12-31",
}


def get_project_root() -> Path:
    """Return repository root (script is in replication/)."""
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir.parent


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path]:
    """Return (similarity_csv_path, controls_csv_path, output_tables_dir)."""
    dc = project_root / "data_clean"
    out_tables = project_root / "outputs" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    return dc / "ecb_similarity_jaccard_bigrams.csv", dc / "controls_month_end.csv", out_tables


def get_window_from_env() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """Return (start_dt, end_dt, start_str, end_str) from env (or defaults)."""
    start_str = os.getenv("ECB_START_DATE", CONFIG["DEFAULT_START_DATE"])
    end_str = os.getenv("ECB_END_DATE", CONFIG["DEFAULT_END_DATE"])
    start_dt, end_dt = pd.Timestamp(start_str), pd.Timestamp(end_str)
    if end_dt < start_dt:
        raise ValueError(f"Invalid window: end < start ({start_str} .. {end_str})")
    return start_dt, end_dt, start_str, end_str


def stars(p: float) -> str:
    """Map p-values to significance stars for compact regression table formatting."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(m, v: str, nd: int = 3) -> str:
    """Format a coefficient with stars (or '.' if the regressor is not in the model)."""
    return f"{m.params[v]:.{nd}f}{stars(float(m.pvalues[v]))}" if v in m.params.index else "."


def ols(df: pd.DataFrame, y: str, xs: list[str]):
    """Fit OLS with HC1 robust standard errors using the selected outcome and regressor list."""
    X = sm.add_constant(df[xs], has_constant="add")
    return sm.OLS(df[y], X, missing="drop").fit(cov_type="HC1")


def load_inputs(sim_path: Path, ctl_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load similarity and controls CSVs and parse date columns."""
    if not sim_path.exists():
        raise FileNotFoundError(f"Missing similarity file: {sim_path} (run step 5 first)")
    if not ctl_path.exists():
        raise FileNotFoundError(f"Missing controls file: {ctl_path} (run step 7b first)")

    sim = pd.read_csv(sim_path)
    ctl = pd.read_csv(ctl_path)
    sim["date"] = pd.to_datetime(sim["date"], errors="coerce")
    ctl["date_m"] = pd.to_datetime(ctl["date_m"], errors="coerce")
    return sim, ctl


def build_regression_df(sim: pd.DataFrame, ctl: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    """Filter similarity to the window, build time indices, merge monthly controls, and return the regression dataframe."""
    df = (
        sim[["date", "sim_jaccard"]]
        .dropna(subset=["date", "sim_jaccard"])
        .loc[lambda d: (d["date"] >= start_dt) & (d["date"] <= end_dt)]
        .query("sim_jaccard > 0")
        .sort_values("date")
        .reset_index(drop=True)
    )

    df["log_similarity"] = np.log(df["sim_jaccard"])
    df["time"] = np.log((df["date"] - start_dt).dt.days + 1)
    df["time_count"] = np.log(np.arange(1, len(df) + 1))

    df["date_m"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.merge(ctl, on="date_m", how="left")


def run_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Run Table 3 specs and return a formatted regression table dataframe."""
    controls = ["output_gap", "inflation", "delta_mro_eom"]

    r1 = ols(df, "log_similarity", controls)
    r2 = ols(df, "log_similarity", ["time"])
    r3 = ols(df, "log_similarity", ["time"] + controls)
    r4 = ols(df, "log_similarity", ["time_count"] + controls)

    def col(m):
        return {
            "Intercept": fmt(m, "const"),
            "Time": fmt(m, "time"),
            "Time (count)": fmt(m, "time_count"),
            "Output gap": fmt(m, "output_gap"),
            "Inflation": fmt(m, "inflation"),
            "Delta MRO": fmt(m, "delta_mro_eom"),
            "Adjusted R2": f"{m.rsquared_adj * 100:.2f}%",
        }

    order = ["Intercept", "Time", "Time (count)", "Output gap", "Inflation", "Delta MRO", "Adjusted R2"]
    return pd.DataFrame({"(1)": col(r1), "(2)": col(r2), "(3)": col(r3), "(4)": col(r4)}).loc[order]


def save_table(table: pd.DataFrame, out_tables: Path) -> Path:
    """Save the regression table CSV to outputs/tables and return the written path."""
    out_path = out_tables / "table3_similarity_regressions.csv"
    table.to_csv(out_path, encoding="utf-8")
    return out_path


def main() -> None:
    """Execute Table 3 regressions and export the formatted table."""
    project_root = get_project_root()
    sim_path, ctl_path, out_tables = resolve_paths(project_root)
    start_dt, end_dt, start_str, end_str = get_window_from_env()

    sim, ctl = load_inputs(sim_path, ctl_path)
    df = build_regression_df(sim, ctl, start_dt, end_dt)

    if len(df) < 5:
        raise ValueError(
            f"Not enough observations in window for regressions after filtering sim_jaccard>0. "
            f"Window={start_str}..{end_str}, n={len(df)}"
        )

    table = run_regressions(df)
    out_path = save_table(table, out_tables)

    print(f"Window (env): {start_str} -> {end_str} | n={len(df)}")
    print(f"Saved: {out_path}")
    print(table.to_string())


if __name__ == "__main__":
    main()