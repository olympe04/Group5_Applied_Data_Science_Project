# 8_regressions-table4.py
# Run Table 4-style OLS regressions for absCAR using pessimism, similarity, and macro controls within an env-defined window.
# I/O:
#   Inputs : data_clean/ecb_similarity_jaccard_bigrams.csv, data_clean/ecb_pessimism_with_car.csv, data_clean/controls_month_end.csv
#   Outputs: outputs/tables/table4_absCAR_regressions.csv
# Notes:
#   The script filters events to the env window, merges monthly controls and similarity, builds log(similarity),
#   runs four OLS specs with HC1 robust SE, and exports a compact regression table (no regression dataset export).

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


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path, Path]:
    """Return (similarity_path, car_path, controls_path, output_tables_dir)."""
    dc = project_root / "data_clean"
    out_tables = project_root / "outputs" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    return (
        dc / "ecb_similarity_jaccard_bigrams.csv",
        dc / "ecb_pessimism_with_car.csv",
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


def load_inputs(sim_path: Path, car_path: Path, ctl_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load similarity, CAR+pessimism, and controls CSVs and parse key date columns."""
    for p, msg in [
        (sim_path, "Missing similarity file (run step 5 first)"),
        (car_path, "Missing CAR+sentiment file (run step 7 first)"),
        (ctl_path, "Missing controls file (run step 7b first)"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{msg}: {p}")

    sim = pd.read_csv(sim_path)
    car = pd.read_csv(car_path)
    ctl = pd.read_csv(ctl_path)

    sim["date"] = pd.to_datetime(sim["date"], errors="coerce")
    car["date"] = pd.to_datetime(car["date"], errors="coerce")
    ctl["date_m"] = pd.to_datetime(ctl["date_m"], errors="coerce")
    return sim, car, ctl


def build_regression_df(
    sim: pd.DataFrame, car: pd.DataFrame, ctl: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> pd.DataFrame:
    """Filter events to window, merge controls and similarity, build log(similarity) and interaction term."""
    car_w = (
        car.dropna(subset=["date"])
           .loc[lambda d: (d["date"] >= start_dt) & (d["date"] <= end_dt)]
           .sort_values("date")
           .reset_index(drop=True)
    )
    if len(car_w) == 0:
        raise ValueError(f"No rows in ecb_pessimism_with_car.csv within window {start_dt.date()}..{end_dt.date()}.")

    df = (
        car_w.assign(date_m=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
             .merge(ctl[["date_m", "output_gap", "inflation", "delta_mro_eom"]], on="date_m", how="left")
             .merge(sim[["date", "sim_jaccard"]], on="date", how="left")
    )

    if "pessimism_lm_pct" not in df.columns:
        raise ValueError("Missing 'pessimism_lm_pct' in ecb_pessimism_with_car.csv (enable ADD_PCT_VERSION in step 6).")
    if "absCAR_pct" not in df.columns:
        raise ValueError("Missing 'absCAR_pct' in ecb_pessimism_with_car.csv (step 7 should create it).")

    df["log_similarity"] = np.where(df["sim_jaccard"] > 0, np.log(df["sim_jaccard"]), np.nan)
    df["pess_x_sim"] = df["pessimism_lm_pct"] * df["log_similarity"]
    return df


def run_regressions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Run Table 4 specs and return (formatted table, n_sim_pos) where n_sim_pos is count of sim>0 rows."""
    controls = ["output_gap", "inflation", "delta_mro_eom"]
    y = "absCAR_pct"

    r1 = ols(df, y, ["pessimism_lm_pct"])
    r2 = ols(df, y, controls)

    dfi = df[df["sim_jaccard"] > 0].copy()
    if len(dfi) == 0:
        raise ValueError("No observations with sim_jaccard > 0; cannot run interaction specs (3)-(4).")

    r3 = ols(dfi, y, ["pess_x_sim"])
    r4 = ols(dfi, y, ["pess_x_sim"] + controls)

    def col(m):
        return {
            "Intercept": fmt(m, "const"),
            "Pessimism": fmt(m, "pessimism_lm_pct"),
            "Pessimism × similarity": fmt(m, "pess_x_sim"),
            "Output gap": fmt(m, "output_gap"),
            "Inflation": fmt(m, "inflation"),
            "Delta MRO": fmt(m, "delta_mro_eom"),
            "Adjusted R²": f"{m.rsquared_adj * 100:.2f}%",
        }

    order = ["Intercept", "Pessimism", "Pessimism × similarity", "Output gap", "Inflation", "Delta MRO", "Adjusted R²"]
    table = pd.DataFrame({"(1)": col(r1), "(2)": col(r2), "(3)": col(r3), "(4)": col(r4)}).loc[order]
    return table, len(dfi)


def save_table(table: pd.DataFrame, out_tables: Path) -> Path:
    """Save the regression table CSV to outputs/tables and return the written path."""
    out_path = out_tables / "table4_absCAR_regressions.csv"
    table.to_csv(out_path, encoding="utf-8")
    return out_path


def main() -> None:
    """Execute Table 4 regressions and export the formatted table."""
    project_root = get_project_root()
    sim_path, car_path, ctl_path, out_tables = resolve_paths(project_root)
    start_dt, end_dt, start_str, end_str = get_window_from_env()

    sim, car, ctl = load_inputs(sim_path, car_path, ctl_path)
    df = build_regression_df(sim, car, ctl, start_dt, end_dt)
    table, n_sim_pos = run_regressions(df)
    out_path = save_table(table, out_tables)

    print(f"Window (env): {start_str} -> {end_str} | n={len(df)} | n(sim>0)={n_sim_pos}")
    print(f"Saved: {out_path}")
    print(table.to_string())


if __name__ == "__main__":
    main()