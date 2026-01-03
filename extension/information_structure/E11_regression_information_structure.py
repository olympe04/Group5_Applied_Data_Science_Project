"""
E11_regression_information_structure.py

Purpose
-------
Estimate whether "information structure" in ECB statements explains market reactions (|CAR|)
beyond similarity, macro controls, and (optionally) tone.

Main modeling choices
---------------------
- Multiple structure proxies are tested because entropy_norm is typically very stable for long texts.
  Proxies:
    * entropy
    * entropy_norm
    * bigrams_unique_ratio
    * ratio_new_tokens
- Similarity is logged: log(sim_jaccard), requiring sim_jaccard > 0.
- Controls merged at month-end; month-end alignment is enforced on BOTH sides.
- OLS with heteroskedasticity-robust SE (HC1).
- Extra robustness tests:
    * Winsorize ratio_new_tokens (1%/99%) and re-estimate key specs
    * Add a "full" spec using bigrams_unique_ratio (in addition to ratio_new_tokens)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DEFAULT_START = "1999-01-01"
DEFAULT_END = "2025-12-31"

CFG = {
    "FEATURES": "data_features/ecb_information_structure.csv",
    "SIM": "data_clean/ecb_similarity_jaccard_bigrams.csv",
    "CAR": "data_clean/ecb_pessimism_with_car.csv",
    "CTL": "data_clean/controls_month_end.csv",
    "OUT": "outputs/tables/table4_absCAR_regressions_info_structure.csv",
}

Y = "absCAR_pct"


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def window() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    s = os.getenv("ECB_START_DATE", DEFAULT_START)
    e = os.getenv("ECB_END_DATE", DEFAULT_END)
    sdt, edt = pd.Timestamp(s), pd.Timestamp(e)
    if edt < sdt:
        raise ValueError(f"Invalid window: {s}..{e}")
    return sdt, edt, s, e


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(m, v: str, nd: int = 3) -> str:
    return f"{m.params[v]:.{nd}f}{stars(float(m.pvalues[v]))}" if v in m.params.index else "."


def month_end(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def winsorize(s: pd.Series, p_lo: float = 0.01, p_hi: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(p_lo), s.quantile(p_hi)
    return s.clip(lower=lo, upper=hi)


def ols_hc1(df: pd.DataFrame, y: str, xs: list[str], min_n: int = 10):
    d = df[[y] + xs].dropna()
    if len(d) < min_n:
        raise ValueError(f"Not enough non-missing observations for OLS: n={len(d)} (need >= {min_n}) | xs={xs}")
    X = sm.add_constant(d[xs], has_constant="add")
    return sm.OLS(d[y], X).fit(cov_type="HC1")


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p_feat = root / CFG["FEATURES"]
    p_sim = root / CFG["SIM"]
    p_car = root / CFG["CAR"]
    p_ctl = root / CFG["CTL"]

    for p in [p_feat, p_sim, p_car, p_ctl]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    feat = pd.read_csv(p_feat, parse_dates=["date"])
    sim = pd.read_csv(p_sim, parse_dates=["date"])
    car = pd.read_csv(p_car, parse_dates=["date"])
    ctl = pd.read_csv(p_ctl, parse_dates=["date_m"])
    return feat, sim, car, ctl


def build_df(
    feat: pd.DataFrame,
    sim: pd.DataFrame,
    car: pd.DataFrame,
    ctl: pd.DataFrame,
    sdt: pd.Timestamp,
    edt: pd.Timestamp,
) -> pd.DataFrame:
    feat = feat.loc[(feat["date"] >= sdt) & (feat["date"] <= edt)].copy()
    sim = sim.loc[(sim["date"] >= sdt) & (sim["date"] <= edt)].copy()
    car = car.loc[(car["date"] >= sdt) & (car["date"] <= edt)].copy()

    if Y not in car.columns:
        raise ValueError(f"Missing '{Y}' in {CFG['CAR']}")
    if "sim_jaccard" not in sim.columns:
        raise ValueError(f"Missing 'sim_jaccard' in {CFG['SIM']}")

    ctl = ctl.copy()
    ctl["date_m"] = month_end(ctl["date_m"])

    df = (
        car.merge(feat, on="date", how="left")
        .merge(sim[["date", "sim_jaccard"]], on="date", how="left")
        .assign(date_m=lambda d: month_end(d["date"]))
        .merge(ctl[["date_m", "output_gap", "inflation", "delta_mro_eom"]], on="date_m", how="left")
    )

    df["log_similarity"] = np.where(df["sim_jaccard"] > 0, np.log(df["sim_jaccard"]), np.nan)

    # robustness: winsorized novelty (does not affect baseline columns)
    if "ratio_new_tokens" in df.columns:
        df["ratio_new_tokens_w"] = winsorize(df["ratio_new_tokens"])

    return df


def run_table(df: pd.DataFrame) -> pd.DataFrame:
    controls = ["output_gap", "inflation", "delta_mro_eom"]
    has_tone = "pessimism_lm_pct" in df.columns

    dfi = df[df["sim_jaccard"] > 0].copy()
    if len(dfi) == 0:
        raise ValueError("No observations with sim_jaccard > 0; cannot run similarity-based specs.")

    models: dict[str, sm.regression.linear_model.RegressionResultsWrapper] = {}

    # Baseline: (structure) + similarity
    structure_vars = [
        ("Entropy + Similarity", ["entropy", "log_similarity"]),
        ("Entropy (norm) + Similarity", ["entropy_norm", "log_similarity"]),
        ("Bigram uniq. ratio + Similarity", ["bigrams_unique_ratio", "log_similarity"]),
        ("New-token ratio + Similarity", ["ratio_new_tokens", "log_similarity"]),
    ]
    for i, (label, xs) in enumerate(structure_vars, start=1):
        xs_ok = [x for x in xs if x in dfi.columns]
        models[f"({i}) {label}"] = ols_hc1(dfi, Y, xs_ok)

    # Full specs (two variants)
    # (5) Full with ratio_new_tokens
    xs5 = ["ratio_new_tokens", "log_similarity"] + controls + (["pessimism_lm_pct"] if has_tone else [])
    models["(5) Full (New-token) + Ctl + Tone"] = ols_hc1(dfi, Y, xs5)

    # (6) Full with bigrams_unique_ratio
    xs6 = ["bigrams_unique_ratio", "log_similarity"] + controls + (["pessimism_lm_pct"] if has_tone else [])
    models["(6) Full (Bigram ratio) + Ctl + Tone"] = ols_hc1(dfi, Y, xs6)

    # Robustness: winsorized novelty
    # (7) New-token ratio (winsor) + Similarity
    if "ratio_new_tokens_w" in dfi.columns:
        models["(7) New-token (winsor) + Similarity"] = ols_hc1(dfi, Y, ["ratio_new_tokens_w", "log_similarity"])

        # (8) Full with winsorized novelty
        xs8 = ["ratio_new_tokens_w", "log_similarity"] + controls + (["pessimism_lm_pct"] if has_tone else [])
        models["(8) Full (winsor New-token) + Ctl + Tone"] = ols_hc1(dfi, Y, xs8)

    def col(m):
        return {
            "Intercept": fmt(m, "const"),
            "Entropy": fmt(m, "entropy"),
            "Entropy (norm)": fmt(m, "entropy_norm"),
            "Bigram uniq. ratio": fmt(m, "bigrams_unique_ratio"),
            "New-token ratio": fmt(m, "ratio_new_tokens"),
            "New-token ratio (winsor)": fmt(m, "ratio_new_tokens_w"),
            "Similarity (log)": fmt(m, "log_similarity"),
            "Pessimism": fmt(m, "pessimism_lm_pct"),
            "Output gap": fmt(m, "output_gap"),
            "Inflation": fmt(m, "inflation"),
            "Delta MRO": fmt(m, "delta_mro_eom"),
            "N": f"{int(m.nobs)}",
            "Adjusted R²": f"{m.rsquared_adj * 100:.2f}%",
        }

    order = [
        "Intercept",
        "Entropy",
        "Entropy (norm)",
        "Bigram uniq. ratio",
        "New-token ratio",
        "New-token ratio (winsor)",
        "Similarity (log)",
        "Pessimism",
        "Output gap",
        "Inflation",
        "Delta MRO",
        "N",
        "Adjusted R²",
    ]

    return pd.DataFrame({k: col(m) for k, m in models.items()}).loc[order]


def main() -> None:
    root = get_project_root()
    sdt, edt, s, e = window()

    feat, sim, car, ctl = load_inputs(root)
    df = build_df(feat, sim, car, ctl, sdt, edt)

    out_table = run_table(df)

    out_path = root / CFG["OUT"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_table.to_csv(out_path, encoding="utf-8")

    n_all = len(df)
    n_pos = int((df["sim_jaccard"] > 0).sum())
    print(f"\nWindow: {s} -> {e} | n={n_all} | n(sim>0)={n_pos}")
    print(f"Saved: {out_path}\n")
    print(out_table.to_string())


if __name__ == "__main__":
    main()
