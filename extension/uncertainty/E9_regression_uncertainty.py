"""
E9_regression_uncertainty.py

Compute Table 4-style OLS regressions for absCAR AND Table 2-style summary statistics in a single script,
for the uncertainty indicator (LM).

Table 4 (regressions):
  Outcome: absCAR_pct
  Specs: (1) indicator only, (2) macro controls only, (3) indicator×log(similarity), (4) interaction + controls
  Robust SE: HC1

Table 2 (summary stats):
  Descriptive statistics for CAR, |CAR|, indicator, similarity, and macro controls (restricted to sim_jaccard > 0).

I/O:
  Inputs :
    - data_clean/ecb_similarity_jaccard_bigrams.csv
    - data_clean/controls_month_end.csv
    - data_features/ecb_uncertainty_with_car.csv
  Outputs:
    - outputs/tables/table4_absCAR_regressions_uncertainty.csv
    - outputs/tables/table2_summary_stats_uncertainty.csv

Window:
  Filters to ECB_START_DATE..ECB_END_DATE from env (defaults: 1999-01-01 .. 2013-12-31).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DEFAULT_START = "1999-01-01"
DEFAULT_END = "2025-12-31"

# Fixed to uncertainty only
IND_COL = "uncertainty_lm_pct"
IND_LABEL = "Uncertainty"


def get_project_root() -> Path:
    """
    Robustly find project root even if this script is moved into subfolders like:
      <root>/extension/uncertainty/E9_regression_uncertainty.py

    Strategy: walk up parents until we find expected root markers.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError(
        "Could not locate project root. Expected to find 'data_clean/' and 'outputs/' in a parent directory."
    )


def get_window_from_env() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """Return (start_dt, end_dt, start_str, end_str) from env."""
    start_str = os.environ.get("ECB_START_DATE", DEFAULT_START)
    end_str = os.environ.get("ECB_END_DATE", DEFAULT_END)
    start_dt, end_dt = pd.Timestamp(start_str), pd.Timestamp(end_str)
    if end_dt < start_dt:
        raise ValueError(f"Invalid window: end < start ({start_str} .. {end_str})")
    return start_dt, end_dt, start_str, end_str


def paths(root: Path) -> tuple[Path, Path, Path, Path]:
    """
    Return (sim_path, event_path, controls_path, out_tables_dir).

    IMPORTANT:
      - similarity (raw Jaccard bigrams file) is read from data_clean/
      - controls_month_end is read from data_clean/
      - event dataset is read from data_features/
    """
    dc = root / "data_clean"
    df = root / "data_features"
    out = root / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    return (
        dc / "ecb_similarity_jaccard_bigrams.csv",  # raw Jaccard file (do not substitute)
        df / "ecb_uncertainty_with_car.csv",
        dc / "controls_month_end.csv",
        out,
    )


def ols_hc1(df: pd.DataFrame, y: str, xs: list[str]):
    """Fit OLS with HC1 robust standard errors."""
    return sm.OLS(df[y], sm.add_constant(df[xs], has_constant="add"), missing="drop").fit(cov_type="HC1")


def stars(p: float) -> str:
    """Map p-values to significance stars."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(m, v: str, nd: int = 3) -> str:
    """Format coefficient with stars (or '.' if absent)."""
    return f"{m.params[v]:.{nd}f}{stars(float(m.pvalues[v]))}" if v in m.params.index else "."


def load(sim_path: Path, event_path: Path, ctl_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load similarity, events, controls and parse date columns."""
    for p, msg in [
        (sim_path, "Missing similarity file (run similarity step first)"),
        (event_path, "Missing event+uncertainty file (run uncertainty event-study step first)"),
        (ctl_path, "Missing controls file (run controls step first)"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{msg}: {p}")

    sim = pd.read_csv(sim_path, parse_dates=["date"])
    evt = pd.read_csv(event_path, parse_dates=["date"])
    ctl = pd.read_csv(ctl_path, parse_dates=["date_m"])
    return sim, evt, ctl


def build_df(
    sim: pd.DataFrame,
    evt: pd.DataFrame,
    ctl: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    """Merge event dataset with similarity and monthly controls inside the window."""
    if "date" not in evt.columns:
        raise ValueError("Event file missing 'date' column.")

    evt_w = evt.loc[lambda d: (d["date"] >= start_dt) & (d["date"] <= end_dt)].copy()
    evt_w["date_m"] = evt_w["date"].dt.to_period("M").dt.to_timestamp()

    if "date_m" not in ctl.columns:
        raise ValueError("Controls file missing 'date_m' column.")
    if "date" not in sim.columns:
        raise ValueError("Similarity file missing 'date' column.")
    if "sim_jaccard" not in sim.columns:
        raise ValueError("Similarity file missing 'sim_jaccard' column.")

    return (
        evt_w.merge(ctl[["date_m", "output_gap", "inflation", "delta_mro_eom"]], on="date_m", how="left")
        .merge(sim[["date", "sim_jaccard"]], on="date", how="left")
    )


def table4(df: pd.DataFrame, ind_col: str, ind_label: str) -> tuple[pd.DataFrame, int]:
    """
    Run Table 4 specs and return (formatted table, n(sim>0)).

    CHANGE:
      - Remove the main effect of similarity (log)
      - Keep only interaction: indicator × log(similarity)
    """
    controls = ["output_gap", "inflation", "delta_mro_eom"]
    y = "absCAR_pct"

    if y not in df.columns:
        raise ValueError(f"Missing '{y}' in merged dataset. (Event-study output should contain it.)")
    if ind_col not in df.columns:
        raise ValueError(f"Missing indicator column '{ind_col}' in event dataset.")
    if "sim_jaccard" not in df.columns:
        raise ValueError("Missing 'sim_jaccard' after merge (check similarity file).")

    df = df.copy()
    df["log_similarity"] = np.where(df["sim_jaccard"] > 0, np.log(df["sim_jaccard"]), np.nan)
    df["ind_x_sim"] = df[ind_col] * df["log_similarity"]

    # (1) indicator only
    r1 = ols_hc1(df, y, [ind_col])
    # (2) controls only
    r2 = ols_hc1(df, y, controls)

    # interaction specs require sim>0
    dfi = df[df["sim_jaccard"] > 0].copy()
    if len(dfi) == 0:
        raise ValueError("No observations with sim_jaccard > 0; cannot run interaction specs (3)-(4).")

    # (3) indicator + interaction ONLY (no log(sim) main effect)
    r3 = ols_hc1(dfi, y, [ind_col, "ind_x_sim"])
    # (4) indicator + interaction + controls
    r4 = ols_hc1(dfi, y, [ind_col, "ind_x_sim"] + controls)

    inter_label = f"{ind_label} × similarity"

    def col(m):
        return {
            "Intercept": fmt(m, "const"),
            ind_label: fmt(m, ind_col),
            inter_label: fmt(m, "ind_x_sim"),
            "Output gap": fmt(m, "output_gap"),
            "Inflation": fmt(m, "inflation"),
            "Delta MRO": fmt(m, "delta_mro_eom"),
            "Adjusted R²": f"{m.rsquared_adj * 100:.2f}%",
        }

    order = [
        "Intercept",
        ind_label,
        inter_label,
        "Output gap",
        "Inflation",
        "Delta MRO",
        "Adjusted R²",
    ]

    out = pd.DataFrame({"(1)": col(r1), "(2)": col(r2), "(3)": col(r3), "(4)": col(r4)}).loc[order]
    return out, len(dfi)


def table2(df: pd.DataFrame, ind_col: str, ind_label: str) -> pd.DataFrame:
    """Compute Table 2-style descriptive statistics on events with sim>0."""
    d = df.query("sim_jaccard > 0").copy()
    if len(d) == 0:
        raise ValueError("No events with sim_jaccard > 0 available for Table 2.")

    cols = [
        ("CAR", "CAR_pct"),
        ("|CAR|", "absCAR_pct"),
        (ind_label, ind_col),
        ("Similarity", "sim_jaccard"),
        ("Output gap", "output_gap"),
        ("Inflation", "inflation"),
        ("Delta MRO", "delta_mro_eom"),
    ]

    for _, c in cols:
        if c not in d.columns:
            raise ValueError(f"Missing column '{c}' required for Table 2.")

    stats = pd.DataFrame(
        {name: pd.to_numeric(d[col], errors="coerce").describe(percentiles=[0.25, 0.5, 0.75]) for name, col in cols}
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


def main() -> None:
    """Run Table 4 and Table 2, print both, and export both CSVs."""
    root = get_project_root()
    start_dt, end_dt, start_str, end_str = get_window_from_env()

    sim_path, event_path, ctl_path, out_dir = paths(root)
    sim, evt, ctl = load(sim_path, event_path, ctl_path)
    df = build_df(sim, evt, ctl, start_dt, end_dt)

    t4, npos = table4(df, IND_COL, IND_LABEL)
    p4 = out_dir / "table4_absCAR_regressions_uncertainty.csv"
    t4.to_csv(p4, encoding="utf-8")

    t2 = table2(df, IND_COL, IND_LABEL)
    p2 = out_dir / "table2_summary_stats_uncertainty.csv"
    t2.to_csv(p2, encoding="utf-8")

    print(f"\nWindow: {start_str} -> {end_str} | n={len(df)} | n(sim>0)={npos}\n")
    print("Table 4 (absCAR regressions):")
    print(t4.to_string())
    print(f"\nSaved: {p4}\n")

    print("Table 2 (summary statistics):")
    print(t2.to_string())
    print(f"\nSaved: {p2}\n")


if __name__ == "__main__":
    main()
