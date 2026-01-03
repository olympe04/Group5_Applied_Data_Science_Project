# E9b_regression_learning_speed.py
# Table 4 regressions using learning-speed similarity computed on BASE Jaccard bigrams:
#   - 1-lag similarity comes from data_clean/ecb_similarity_jaccard_bigrams.csv (sim_jaccard)
#   - centroid-k similarity comes from data_features/ecb_similarity_jaccard_learning.csv (sim_centroid_k)
#
# IMPORTANT: Specs (3)-(4) are INTERACTION-ONLY (paper-style), i.e. only the interaction term enters the model.
# Exports ONLY regression tables.
#
# I/O:
#   Inputs :
#     - data_features/ecb_similarity_jaccard_learning.csv
#     - data_clean/ecb_similarity_jaccard_bigrams.csv
#     - data_clean/ecb_pessimism_with_car.csv
#     - data_clean/controls_month_end.csv
#   Outputs:
#     - outputs/tables/table4_absCAR_regressions_centroidk_jaccard.csv
#     - outputs/tables/table4_absCAR_regressions_centroidk_jaccard_z.csv

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


CFG = {
    "START_DEFAULT": "1999-01-01",
    "END_DEFAULT": "2013-12-31",

    # centroid-k learning-speed file (from your Jaccard learning-speed step)
    "LEARNING_FILE": "data_features/ecb_similarity_jaccard_learning.csv",

    # base 1-lag Jaccard series (from replication step 5)
    "BASE_SIM_FILE": "data_clean/ecb_similarity_jaccard_bigrams.csv",

    "CAR_FILE": "data_clean/ecb_pessimism_with_car.csv",
    "CTL_FILE": "data_clean/controls_month_end.csv",
    "OUT_DIR": "outputs/tables",
}

T4_LOG_ORDER = [
    "Intercept", "Pessimism", "Similarity (log)", "Pessimism × similarity",
    "Output gap", "Inflation", "Delta MRO", "Adjusted R²",
]
T4_Z_ORDER = [
    "Intercept", "Pessimism", "Similarity (z)", "Pessimism × similarity",
    "Output gap", "Inflation", "Delta MRO", "Adjusted R²",
]


def get_project_root() -> Path:
    """
    Robustly find project root even if this script is in:
      <root>/extension/learning_speed/E9b_regression_jaccard_learning_speed.py
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError(
        "Could not locate project root. Expected to find 'data_clean/' and 'outputs/' in a parent directory."
    )


def window() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    s = os.getenv("ECB_START_DATE", CFG["START_DEFAULT"])
    e = os.getenv("ECB_END_DATE", CFG["END_DEFAULT"])
    sdt, edt = pd.Timestamp(s), pd.Timestamp(e)
    if edt < sdt:
        raise ValueError(f"Invalid window: {s}..{e}")
    return sdt, edt, s, e


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(m, v: str, nd: int = 3) -> str:
    return f"{m.params[v]:.{nd}f}{stars(float(m.pvalues[v]))}" if v in m.params.index else "."


def ols(df: pd.DataFrame, y: str, xs: list[str]):
    X = sm.add_constant(df[xs], has_constant="add")
    return sm.OLS(df[y], X, missing="drop").fit(cov_type="HC1")


def to_date(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce")


def save_table(df: pd.DataFrame, path: Path, order: list[str]) -> None:
    df.reindex(order).to_csv(path, encoding="utf-8")


def base_paths() -> tuple[Path, Path]:
    project_root = get_project_root()
    out = project_root / CFG["OUT_DIR"]
    out.mkdir(parents=True, exist_ok=True)
    return project_root, out


def load_inputs(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      learning: (date, sim_centroid_k)
      base    : (date, sim_jaccard)
      car     : CAR+pessimism dataset
      ctl     : monthly controls
    """
    p_learn = project_root / CFG["LEARNING_FILE"]
    p_base = project_root / CFG["BASE_SIM_FILE"]
    p_car = project_root / CFG["CAR_FILE"]
    p_ctl = project_root / CFG["CTL_FILE"]

    for p, msg in [
        (p_learn, "Missing learning-speed file (run Jaccard learning-speed step first)"),
        (p_base, "Missing base Jaccard similarity file (run replication step 5 first)"),
        (p_car, "Missing CAR+pessimism file (run replication steps 6+7 first)"),
        (p_ctl, "Missing controls file (run replication step 7b first)"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{msg}: {p}")

    learning = pd.read_csv(p_learn)
    base = pd.read_csv(p_base)
    car = pd.read_csv(p_car)
    ctl = pd.read_csv(p_ctl)

    learning["date"] = to_date(learning["date"])
    base["date"] = to_date(base["date"])
    car["date"] = to_date(car["date"])
    ctl["date_m"] = to_date(ctl["date_m"])

    if "sim_centroid_k" not in learning.columns:
        raise ValueError("Missing 'sim_centroid_k' in learning-speed file. (Run the learning-speed step.)")
    if "sim_jaccard" not in base.columns:
        raise ValueError("Missing 'sim_jaccard' in base similarity file (expected ecb_similarity_jaccard_bigrams.csv).")
    if "absCAR_pct" not in car.columns:
        raise ValueError("Missing 'absCAR_pct' in CAR file.")
    if "pessimism_lm_pct" not in car.columns:
        raise ValueError("Missing 'pessimism_lm_pct' in CAR file (step 6 should create it).")

    learning = learning[["date", "sim_centroid_k"]].dropna(subset=["date", "sim_centroid_k"])
    base = base[["date", "sim_jaccard"]].dropna(subset=["date", "sim_jaccard"])
    car = car.dropna(subset=["date"])

    return learning, base, car, ctl


def run_table4_centroidk_interaction_only(
    car: pd.DataFrame,
    learning: pd.DataFrame,
    base: pd.DataFrame,
    ctl: pd.DataFrame,
    out: Path,
    sdt: pd.Timestamp,
    edt: pd.Timestamp,
    s: str,
    e: str,
) -> None:
    """
    Table 4 with centroid-k similarity:
      (1) Pessimism only
      (2) Controls only
      (3) Interaction-only: pess × log(sim_centroid_k)
      (4) Interaction-only + controls

    Also exports z(sim_centroid_k) variants (3z)-(4z).
    """
    controls = ["output_gap", "inflation", "delta_mro_eom"]
    y = "absCAR_pct"

    car_w = (
        car.loc[(car["date"] >= sdt) & (car["date"] <= edt)]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(car_w) == 0:
        raise ValueError(f"No CAR rows in window {s}..{e}")

    df = (
        car_w.assign(date_m=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .merge(ctl[["date_m"] + controls], on="date_m", how="left")
        .merge(learning, on="date", how="left")            # sim_centroid_k
        .merge(base, on="date", how="left")                # sim_jaccard (base 1-lag, kept for reference)
    )

    # -------- LOG(sim_centroid_k) + interaction-only --------
    dfl = df.query("sim_centroid_k > 0").copy()
    if len(dfl) == 0:
        raise ValueError(f"[Table4 centroid-k log] No observations with sim_centroid_k > 0 in {s}..{e}")

    dfl["log_similarity"] = np.log(dfl["sim_centroid_k"])
    dfl["pess_x_logsim"] = dfl["pessimism_lm_pct"] * dfl["log_similarity"]

    # EXACT paper-style specs:
    r1 = ols(df, y, ["pessimism_lm_pct"])
    r2 = ols(df, y, controls)
    r3 = ols(dfl, y, ["pess_x_logsim"])
    r4 = ols(dfl, y, ["pess_x_logsim"] + controls)

    def col_log(m):
        return {
            "Intercept": fmt(m, "const"),
            "Pessimism": fmt(m, "pessimism_lm_pct"),
            "Similarity (log)": fmt(m, "log_similarity"),
            "Pessimism × similarity": fmt(m, "pess_x_logsim"),
            "Output gap": fmt(m, "output_gap"),
            "Inflation": fmt(m, "inflation"),
            "Delta MRO": fmt(m, "delta_mro_eom"),
            "Adjusted R²": f"{m.rsquared_adj * 100:.2f}%",
        }

    table_log = pd.DataFrame({"(1)": col_log(r1), "(2)": col_log(r2), "(3)": col_log(r3), "(4)": col_log(r4)})
    save_table(table_log, out / "table4_absCAR_regressions_centroidk_jaccard.csv", T4_LOG_ORDER)

    print(f"[Table4 centroid-k Jaccard log | interaction-only] Window: {s}->{e} | n={len(df)} | n(sim>0)={len(dfl)}")
    print(table_log.reindex(T4_LOG_ORDER).to_string())

    # -------- Z(sim_centroid_k) + interaction-only --------
    dfz = df.dropna(subset=["sim_centroid_k"]).copy()
    ssim = dfz["sim_centroid_k"].astype(float)
    sd = ssim.std(ddof=1)
    if len(dfz) == 0 or not sd or np.isnan(sd) or sd <= 0:
        print("[Table4 centroid-k Jaccard z] skipped (no variation in sim_centroid_k)")
        return

    dfz["z_sim"] = (ssim - ssim.mean()) / sd
    dfz["pess_x_zsim"] = dfz["pessimism_lm_pct"] * dfz["z_sim"]

    rz3 = ols(dfz, y, ["pess_x_zsim"])
    rz4 = ols(dfz, y, ["pess_x_zsim"] + controls)

    def col_z(m):
        return {
            "Intercept": fmt(m, "const"),
            "Pessimism": fmt(m, "pessimism_lm_pct"),
            "Similarity (z)": fmt(m, "z_sim"),
            "Pessimism × similarity": fmt(m, "pess_x_zsim"),
            "Output gap": fmt(m, "output_gap"),
            "Inflation": fmt(m, "inflation"),
            "Delta MRO": fmt(m, "delta_mro_eom"),
            "Adjusted R²": f"{m.rsquared_adj * 100:.2f}%",
        }

    table_z = pd.DataFrame({"(3z)": col_z(rz3), "(4z)": col_z(rz4)})
    save_table(table_z, out / "table4_absCAR_regressions_centroidk_jaccard_z.csv", T4_Z_ORDER)

    print(f"[Table4 centroid-k Jaccard z | interaction-only] Window: {s}->{e} | n={len(dfz)}")
    print(table_z.reindex(T4_Z_ORDER).to_string())


def main() -> None:
    project_root, out = base_paths()
    sdt, edt, s, e = window()
    learning, base, car, ctl = load_inputs(project_root)
    run_table4_centroidk_interaction_only(car, learning, base, ctl, out, sdt, edt, s, e)


if __name__ == "__main__":
    main()
