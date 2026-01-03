# E8_regressions-table4_asymmetry.py
# Extension Table 4: asymmetry "bad news vs good news" + interactions with similarity.
#
# Improvements included:
# 1) Make "good news" positive for easier interpretation:
#       good_news_lm_pct = -pess_pos_lm_pct  (>=0)
#       good_x_sim = good_news_lm_pct * log_similarity_c
#    Then a positive coefficient means "more good news => absCAR increases" (or decreases).
#
# 2) Wald tests for asymmetry (joint models):
#       H0: beta_bad = beta_good
#       H0: beta_bad×sim = beta_good×sim
#
# Notes on centering:
# - sim in (0,1) => log(sim) is negative, behaves like a constant -> collinearity risk in interactions.
# - Center log(sim): log_sim_c = log_sim - mean(log_sim)
#   => level coeff = effect at average similarity; interaction = change with similarity deviations.

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


CONFIG = {
    "DEFAULT_START_DATE": "1999-01-01",
    "DEFAULT_END_DATE": "2013-12-31",
    # Choose which joint spec to use for Wald tests: 3 or 4
    "WALD_ON_SPEC": 4,
}


def get_project_root() -> Path:
    """
    Robustly find project root even if this script is moved into subfolders like:
      <root>/extension/separate_goodnews_badnews/E8_regressions-table4_asymmetry.py

    Strategy: walk up parents until we find expected root markers.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError(
        "Could not locate project root. Expected to find 'data_clean/' and 'outputs/' in a parent directory."
    )


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path, Path]:
    dc = project_root / "data_clean"
    out_tables = project_root / "outputs" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    return (
        dc / "ecb_similarity_jaccard_bigrams.csv",
        dc / "ecb_pessimism_with_car.csv",   # <- back to data_clean (file is here)
        dc / "controls_month_end.csv",
        out_tables,
    )



def get_window_from_env() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    start_str = os.getenv("ECB_START_DATE", CONFIG["DEFAULT_START_DATE"])
    end_str = os.getenv("ECB_END_DATE", CONFIG["DEFAULT_END_DATE"])
    start_dt, end_dt = pd.Timestamp(start_str), pd.Timestamp(end_str)
    if end_dt < start_dt:
        raise ValueError(f"Invalid window: end < start ({start_str} .. {end_str})")
    return start_dt, end_dt, start_str, end_str


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(m, v: str, nd: int = 3) -> str:
    return f"{m.params[v]:.{nd}f}{stars(float(m.pvalues[v]))}" if v in m.params.index else "."


def ols(df: pd.DataFrame, y: str, xs: list[str]):
    X = sm.add_constant(df[xs], has_constant="add")
    return sm.OLS(df[y], X, missing="drop").fit(cov_type="HC1")


def load_inputs(sim_path: Path, car_path: Path, ctl_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    car_w = (
        car.dropna(subset=["date"])
        .loc[lambda d: (d["date"] >= start_dt) & (d["date"] <= end_dt)]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(car_w) == 0:
        raise ValueError(
            f"No rows in ecb_pessimism_with_car.csv within window {start_dt.date()}..{end_dt.date()}."
        )

    required = ["absCAR_pct", "pess_neg_lm_pct", "pess_pos_lm_pct"]
    missing = [c for c in required if c not in car_w.columns]
    if missing:
        raise ValueError(
            "Missing required columns in ecb_pessimism_with_car.csv: "
            + ", ".join(missing)
            + "\nTip: run E6_sentiment_pessimism_lm.py then run step 7 again."
        )

    df = (
        car_w.assign(date_m=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .merge(ctl[["date_m", "output_gap", "inflation", "delta_mro_eom"]], on="date_m", how="left")
        .merge(sim[["date", "sim_jaccard"]], on="date", how="left")
    )

    # log(sim) defined when sim>0
    df["log_similarity"] = np.where(df["sim_jaccard"] > 0, np.log(df["sim_jaccard"]), np.nan)

    # Center log(sim) to reduce multicollinearity in interaction models
    df["log_similarity_c"] = df["log_similarity"] - df["log_similarity"].mean(skipna=True)

    # Bad news: already positive by construction (>=0)
    # Good news: make it positive for easy reading (>=0)
    df["bad_news_lm_pct"] = df["pess_neg_lm_pct"]
    df["good_news_lm_pct"] = -df["pess_pos_lm_pct"]  # flip sign => positive intensity of good news

    # Interactions with centered similarity
    df["bad_x_sim"] = df["bad_news_lm_pct"] * df["log_similarity_c"]
    df["good_x_sim"] = df["good_news_lm_pct"] * df["log_similarity_c"]

    return df


def wald_tests_joint(model, label: str) -> None:
    """
    Print Wald tests for asymmetry:
      H0: bad_news = good_news
      H0: bad_x_sim = good_x_sim
    """
    tests = [
        ("H0: beta_bad = beta_good", "bad_news_lm_pct = good_news_lm_pct"),
        ("H0: beta_bad×sim = beta_good×sim", "bad_x_sim = good_x_sim"),
    ]
    print(f"\nWald tests ({label}):")
    for name, hyp in tests:
        try:
            wt = model.wald_test(hyp, scalar=True)
            stat = float(np.asarray(wt.statistic).ravel()[0])
            pval = float(np.asarray(wt.pvalue).ravel()[0])
            print(f"  {name}: stat={stat:.3f}, p={pval:.4f}")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")


def run_regressions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    controls = ["output_gap", "inflation", "delta_mro_eom"]
    y = "absCAR_pct"

    # (1)-(2): levels only (bad + good)
    r1 = ols(df, y, ["bad_news_lm_pct", "good_news_lm_pct"])
    r2 = ols(df, y, ["bad_news_lm_pct", "good_news_lm_pct"] + controls)

    # Interaction specs require sim>0
    dfi = df[df["sim_jaccard"] > 0].copy()
    if len(dfi) == 0:
        raise ValueError("No observations with sim_jaccard > 0; cannot run interaction specs.")

    # (3)-(4): joint model (levels + interactions)
    r3 = ols(dfi, y, ["bad_news_lm_pct", "good_news_lm_pct", "bad_x_sim", "good_x_sim"])
    r4 = ols(dfi, y, ["bad_news_lm_pct", "good_news_lm_pct", "bad_x_sim", "good_x_sim"] + controls)

    # (5)-(6): bad only (level + interaction)
    r5 = ols(dfi, y, ["bad_news_lm_pct", "bad_x_sim"])
    r6 = ols(dfi, y, ["bad_news_lm_pct", "bad_x_sim"] + controls)

    # (7)-(8): good only (level + interaction)
    r7 = ols(dfi, y, ["good_news_lm_pct", "good_x_sim"])
    r8 = ols(dfi, y, ["good_news_lm_pct", "good_x_sim"] + controls)

    # Wald tests on the joint model (spec 3 or 4)
    if int(CONFIG.get("WALD_ON_SPEC", 4)) == 3:
        wald_tests_joint(r3, "spec (3) Levels + Int")
    else:
        wald_tests_joint(r4, "spec (4) Levels + Int + Controls")

    def col(m):
        return {
            "Intercept": fmt(m, "const"),
            "Bad news (>=0)": fmt(m, "bad_news_lm_pct"),
            "Good news (>=0)": fmt(m, "good_news_lm_pct"),
            "Bad news × similarity (centered)": fmt(m, "bad_x_sim"),
            "Good news × similarity (centered)": fmt(m, "good_x_sim"),
            "Output gap": fmt(m, "output_gap"),
            "Inflation": fmt(m, "inflation"),
            "Delta MRO": fmt(m, "delta_mro_eom"),
            "Adjusted R²": f"{m.rsquared_adj * 100:.2f}%",
        }

    order = [
        "Intercept",
        "Bad news (>=0)",
        "Good news (>=0)",
        "Bad news × similarity (centered)",
        "Good news × similarity (centered)",
        "Output gap",
        "Inflation",
        "Delta MRO",
        "Adjusted R²",
    ]

    table = pd.DataFrame(
        {
            "(1) Levels": col(r1),
            "(2) Levels + Ctls": col(r2),
            "(3) Levels + Int": col(r3),
            "(4) Levels + Int + Ctls": col(r4),
            "(5) Bad only + Int": col(r5),
            "(6) Bad only + Int + Ctls": col(r6),
            "(7) Good only + Int": col(r7),
            "(8) Good only + Int + Ctls": col(r8),
        }
    ).loc[order]

    return table, len(dfi)


def save_table(table: pd.DataFrame, out_tables: Path) -> Path:
    out_path = out_tables / "Etable4_absCAR_regressions_asymmetry.csv"
    table.to_csv(out_path, encoding="utf-8")
    return out_path


def main() -> None:
    project_root = get_project_root()
    sim_path, car_path, ctl_path, out_tables = resolve_paths(project_root)
    start_dt, end_dt, start_str, end_str = get_window_from_env()

    sim, car, ctl = load_inputs(sim_path, car_path, ctl_path)
    df = build_regression_df(sim, car, ctl, start_dt, end_dt)

    table, n_sim_pos = run_regressions(df)
    out_path = save_table(table, out_tables)

    print(f"\nWindow (env): {start_str} -> {end_str} | n={len(df)} | n(sim>0)={n_sim_pos}")
    print(f"Saved: {out_path}")
    print(table.to_string())


if __name__ == "__main__":
    main()
