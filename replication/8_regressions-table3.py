# 8_regressions-table3.py
# Estimate Table 3-style OLS regressions for log(similarity) within an env-defined observation window.
# I/O:
#   Inputs: data_clean/ecb_similarity_jaccard_bigrams.csv and data_clean/controls_month_end.csv.
#   Outputs: outputs/table3_similarity_regressions.csv and outputs/regression_dataset_table3.csv.

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


def main() -> None:
    """Load similarity and controls, build time indices, run Table 3 regressions, and export results to CSV."""
    scripts_dir = Path(__file__).resolve().parent   # .../replication
    project_root = scripts_dir.parent               # .../ (repo root)

    dc = project_root / "data_clean"
    out = project_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    start_dt, end_dt, start_str, end_str = get_window_from_env()

    sim_path = dc / "ecb_similarity_jaccard_bigrams.csv"
    ctl_path = dc / "controls_month_end.csv"
    if not sim_path.exists():
        raise FileNotFoundError(f"Missing similarity file: {sim_path} (run step 5 first)")
    if not ctl_path.exists():
        raise FileNotFoundError(f"Missing controls file: {ctl_path} (run step 7b first)")

    sim = pd.read_csv(sim_path)
    ctl = pd.read_csv(ctl_path)

    sim["date"] = pd.to_datetime(sim["date"], errors="coerce")
    ctl["date_m"] = pd.to_datetime(ctl["date_m"], errors="coerce")

    df = (
        sim[["date", "sim_jaccard"]]
        .dropna(subset=["date", "sim_jaccard"])
        .loc[lambda d: (d["date"] >= start_dt) & (d["date"] <= end_dt)]
        .query("sim_jaccard > 0")
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(df) < 5:
        raise ValueError(
            f"Not enough observations in window for regressions after filtering sim_jaccard>0. "
            f"Window={start_str}..{end_str}, n={len(df)}"
        )

    df["log_similarity"] = np.log(df["sim_jaccard"])
    df["time"] = np.log((df["date"] - start_dt).dt.days + 1)
    df["time_count"] = np.log(np.arange(1, len(df) + 1))

    df["date_m"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.merge(ctl, on="date_m", how="left")

    infl = "inflation"
    dmro = "delta_mro_eom"
    controls = ["output_gap", infl, dmro]

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
            "Inflation": fmt(m, infl),
            "Delta MRO": fmt(m, dmro),
            "Adjusted R2": f"{m.rsquared_adj * 100:.2f}%",
        }

    table = pd.DataFrame({"(1)": col(r1), "(2)": col(r2), "(3)": col(r3), "(4)": col(r4)}).loc[
        ["Intercept", "Time", "Time (count)", "Output gap", "Inflation", "Delta MRO", "Adjusted R2"]
    ]

    table.to_csv(out / "table3_similarity_regressions.csv", encoding="utf-8")

    keep = ["date", "date_m", "sim_jaccard", "log_similarity", "time", "time_count"] + controls
    df_out = df[keep].copy()
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    df_out["date_m"] = df_out["date_m"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(out / "regression_dataset_table3.csv", index=False, encoding="utf-8")

    print(f"Window (env): {start_str} -> {end_str} | n={len(df)}")
    print(table.to_string())


if __name__ == "__main__":
    main()
