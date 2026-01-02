# 8_regressions-table4.py
# Run Table 4-style OLS regressions for absCAR using pessimism, similarity, and macro controls within an env-defined window.
# I/O:
#   Inputs: data_clean/ecb_similarity_jaccard_bigrams.csv, data_clean/ecb_pessimism_with_car.csv, data_clean/controls_month_end.csv.
#   Outputs: outputs/table4_absCAR_regressions.csv and outputs/regression_dataset_table4.csv.

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


def get_window_from_env() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """Read (start, end) regression window from env (or defaults) and return parsed timestamps plus the original strings."""
    start_str = os.getenv("ECB_START_DATE", CONFIG["DEFAULT_START_DATE"])
    end_str = os.getenv("ECB_END_DATE", CONFIG["DEFAULT_END_DATE"])
    start_dt = pd.Timestamp(start_str)
    end_dt = pd.Timestamp(end_str)
    if end_dt < start_dt:
        raise ValueError(f"Invalid window: end < start ({start_str} .. {end_str})")
    return start_dt, end_dt, start_str, end_str


def stars(p):
    """Map p-values to significance stars for compact regression table formatting."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(m, v, nd=3):
    """Format a coefficient with stars (or '.' if the regressor is not in the model)."""
    return f"{m.params[v]:.{nd}f}{stars(m.pvalues[v])}" if v in m.params.index else "."


def ols(df, y, xs):
    """Fit OLS with HC1 robust standard errors using the selected outcome and regressor list."""
    X = sm.add_constant(df[xs], has_constant="add")
    return sm.OLS(df[y], X, missing="drop").fit(cov_type="HC1")


def main() -> None:
    """Load inputs, restrict to window, estimate specs (1)-(4), and export the regression table and dataset."""
    scripts_dir = Path(__file__).resolve().parent   # .../replication
    project_root = scripts_dir.parent               # .../ (repo root)

    dc = project_root / "data_clean"
    out = project_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    start_dt, end_dt, start_str, end_str = get_window_from_env()

    sim_path = dc / "ecb_similarity_jaccard_bigrams.csv"
    car_path = dc / "ecb_pessimism_with_car.csv"
    ctl_path = dc / "controls_month_end.csv"

    if not sim_path.exists():
        raise FileNotFoundError(f"Missing similarity file: {sim_path} (run step 5 first)")
    if not car_path.exists():
        raise FileNotFoundError(f"Missing CAR+sentiment file: {car_path} (run step 7 first)")
    if not ctl_path.exists():
        raise FileNotFoundError(f"Missing controls file: {ctl_path} (run step 7b first)")

    sim = pd.read_csv(sim_path)
    car = pd.read_csv(car_path)
    ctl = pd.read_csv(ctl_path)

    sim["date"] = pd.to_datetime(sim["date"], errors="coerce")
    car["date"] = pd.to_datetime(car["date"], errors="coerce")
    ctl["date_m"] = pd.to_datetime(ctl["date_m"], errors="coerce")

    car_w = (
        car.dropna(subset=["date"])
           .loc[lambda d: (d["date"] >= start_dt) & (d["date"] <= end_dt)]
           .sort_values("date")
           .reset_index(drop=True)
    )
    if len(car_w) == 0:
        raise ValueError(f"No rows in ecb_pessimism_with_car.csv within window {start_str}..{end_str}.")

    df = (
        car_w
        .assign(date_m=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .merge(ctl[["date_m", "output_gap", "inflation", "delta_mro_eom"]], on="date_m", how="left")
        .merge(sim[["date", "sim_jaccard"]], on="date", how="left")
    )

    df["log_similarity"] = np.where(df["sim_jaccard"] > 0, np.log(df["sim_jaccard"]), np.nan)

    if "pessimism_lm_pct" not in df.columns:
        raise ValueError("Missing 'pessimism_lm_pct' in ecb_pessimism_with_car.csv (enable ADD_PCT_VERSION in step 6).")
    if "absCAR_pct" not in df.columns:
        raise ValueError("Missing 'absCAR_pct' in ecb_pessimism_with_car.csv (step 7 should create it).")

    df["pess_x_sim"] = df["pessimism_lm_pct"] * df["log_similarity"]

    controls = ["output_gap", "inflation", "delta_mro_eom"]
    y = "absCAR_pct"

    r1 = ols(df, y, ["pessimism_lm_pct"])
    r2 = ols(df, y, controls)

    dfi = df[df["sim_jaccard"] > 0].copy()
    if len(dfi) == 0:
        raise ValueError(
            f"No observations with sim_jaccard > 0 in window {start_str}..{end_str}; "
            "cannot run interaction specs (3)-(4)."
        )

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
            "Adjusted R²": f"{m.rsquared_adj*100:.2f}%",
        }

    table = pd.DataFrame({"(1)": col(r1), "(2)": col(r2), "(3)": col(r3), "(4)": col(r4)}).loc[
        ["Intercept", "Pessimism", "Pessimism × similarity", "Output gap", "Inflation", "Delta MRO", "Adjusted R²"]
    ]

    table.to_csv(out / "table4_absCAR_regressions.csv", encoding="utf-8")

    keep = [
        "date", "date_m", "absCAR_pct", "pessimism_lm_pct", "sim_jaccard",
        "log_similarity", "pess_x_sim",
    ] + controls

    df_out = df[keep].copy()
    df_out["date"] = pd.to_datetime(df_out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_out["date_m"] = pd.to_datetime(df_out["date_m"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_out.to_csv(out / "regression_dataset_table4.csv", index=False, encoding="utf-8")

    print(f"Window (env): {start_str} -> {end_str} | n={len(df)} | n(sim>0)={len(dfi)}")
    print(table.to_string())


if __name__ == "__main__":
    main()
