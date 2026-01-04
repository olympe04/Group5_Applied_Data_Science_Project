# E9_regression_uncertainty.py
# Table 4-style regressions (absCAR_pct) + Table 2-style summary stats (uncertainty LM)

from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_START = "1999-01-01"
DEFAULT_END = "2013-12-31"

IND_COL = "uncertainty_lm_pct"
IND_LABEL = "Uncertainty"


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError("Project root not found (expected data_clean/ and outputs/).")


def get_window() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    s = os.environ.get("ECB_START_DATE", DEFAULT_START)
    e = os.environ.get("ECB_END_DATE", DEFAULT_END)
    sdt, edt = pd.Timestamp(s), pd.Timestamp(e)
    if edt < sdt:
        raise ValueError(f"Invalid window: {s}..{e}")
    return sdt, edt, s, e


def paths(root: Path):
    dc, df = root / "data_clean", root / "data_features"
    out = root / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    return (
        dc / "ecb_similarity_jaccard_bigrams.csv",
        df / "ecb_uncertainty_with_car.csv",
        dc / "controls_month_end.csv",
        out,
    )


def ols_hc1(d: pd.DataFrame, y: str, xs: list[str]):
    X = sm.add_constant(d[xs], has_constant="add")
    return sm.OLS(d[y], X, missing="drop").fit(cov_type="HC1")


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(m, v: str, nd: int = 3) -> str:
    if v not in m.params.index:
        return "."
    return f"{m.params[v]:.{nd}f}{stars(float(m.pvalues[v]))}"


def load(sim_path: Path, evt_path: Path, ctl_path: Path):
    for p in [sim_path, evt_path, ctl_path]:
        if not p.exists():
            raise FileNotFoundError(p)
    sim = pd.read_csv(sim_path, parse_dates=["date"])
    evt = pd.read_csv(evt_path, parse_dates=["date"])
    ctl = pd.read_csv(ctl_path, parse_dates=["date_m"])
    return sim, evt, ctl


def build_df(sim: pd.DataFrame, evt: pd.DataFrame, ctl: pd.DataFrame, sdt, edt) -> pd.DataFrame:
    evt = evt.loc[(evt["date"] >= sdt) & (evt["date"] <= edt)].copy()
    evt["date_m"] = evt["date"].dt.to_period("M").dt.to_timestamp()

    if "sim_jaccard" not in sim.columns:
        raise ValueError("Similarity file must contain 'sim_jaccard'.")
    need_ctl = ["date_m", "output_gap", "inflation", "delta_mro_eom"]
    for c in need_ctl:
        if c not in ctl.columns:
            raise ValueError(f"Controls file missing '{c}'.")

    return (
        evt.merge(ctl[need_ctl], on="date_m", how="left")
           .merge(sim[["date", "sim_jaccard"]], on="date", how="left")
    )


def table4(df: pd.DataFrame, ind_col: str, ind_label: str):
    y = "absCAR_pct"
    controls = ["output_gap", "inflation", "delta_mro_eom"]
    if y not in df.columns or ind_col not in df.columns:
        raise ValueError(f"Need columns '{y}' and '{ind_col}' in merged dataset.")

    d = df.copy()
    d["log_similarity"] = np.where(d["sim_jaccard"] > 0, np.log(d["sim_jaccard"]), np.nan)
    d["ind_x_logsim"] = d[ind_col] * d["log_similarity"]

    r1 = ols_hc1(d, y, [ind_col])
    r2 = ols_hc1(d, y, controls)

    di = d.query("sim_jaccard > 0").copy()
    if di.empty:
        raise ValueError("No observations with sim_jaccard > 0 for interaction specs.")

    # ✅ (3) and (4): DROP standalone Uncertainty, keep ONLY interaction
    r3 = ols_hc1(di, y, ["ind_x_logsim"])
    r4 = ols_hc1(di, y, ["ind_x_logsim"] + controls)

    inter = f"{ind_label} × log(Similarity)"

    def col(m):
        return {
            "Intercept": fmt(m, "const"),
            ind_label: fmt(m, ind_col),                 # will be "." in (3)(4)
            inter: fmt(m, "ind_x_logsim"),
            "Output gap": fmt(m, "output_gap"),
            "Inflation": fmt(m, "inflation"),
            "Delta MRO": fmt(m, "delta_mro_eom"),
            "Adjusted R²": f"{m.rsquared_adj * 100:.2f}%",
        }

    order = ["Intercept", ind_label, inter, "Output gap", "Inflation", "Delta MRO", "Adjusted R²"]
    out = pd.DataFrame({"(1)": col(r1), "(2)": col(r2), "(3)": col(r3), "(4)": col(r4)}).loc[order]
    return out, len(di)


def table2(df: pd.DataFrame, ind_col: str, ind_label: str) -> pd.DataFrame:
    d = df.query("sim_jaccard > 0").copy()
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
            raise ValueError(f"Missing '{c}' for Table 2.")

    stats = pd.DataFrame(
        {name: pd.to_numeric(d[col], errors="coerce").describe(percentiles=[0.25, 0.5, 0.75]) for name, col in cols}
    ).T.rename(
        columns={"mean": "Mean", "std": "Std. dev.", "min": "Min.", "25%": "Quartile 1",
                 "50%": "Median", "75%": "Quartile 3", "max": "Max."}
    )[["Mean", "Std. dev.", "Min.", "Quartile 1", "Median", "Quartile 3", "Max."]].round(2)
    return stats


def main() -> None:
    root = get_project_root()
    sdt, edt, s, e = get_window()
    sim_path, evt_path, ctl_path, out_dir = paths(root)

    sim, evt, ctl = load(sim_path, evt_path, ctl_path)
    df = build_df(sim, evt, ctl, sdt, edt)

    t4, npos = table4(df, IND_COL, IND_LABEL)
    t2 = table2(df, IND_COL, IND_LABEL)

    p4 = out_dir / "table4_absCAR_regressions_uncertainty.csv"
    p2 = out_dir / "table2_summary_stats_uncertainty.csv"
    t4.to_csv(p4, encoding="utf-8")
    t2.to_csv(p2, encoding="utf-8")

    print(f"\nWindow: {s} -> {e} | n={len(df)} | n(sim>0)={npos}\n")
    print("Table 4:")
    print(t4.to_string())
    print(f"\nSaved: {p4}\n")
    print("Table 2:")
    print(t2.to_string())
    print(f"\nSaved: {p2}\n")


if __name__ == "__main__":
    main()