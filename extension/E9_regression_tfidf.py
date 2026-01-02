# E9_tfidf_tables_concise.py
# Table 3 + Table 4 (log + z) using TF-IDF cosine similarity in one concise script.

from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


CFG = {
    "START_DEFAULT": "1999-01-01",
    "END_DEFAULT": "2025-12-31",
    "SIM_FILE": "data_features/ecb_similarity_tfidf.csv",
    "CAR_FILE": "data_clean/ecb_pessimism_with_car.csv",
    "CTL_FILE": "data_clean/controls_month_end.csv",
    "OUT_DIR": "outputs",
}


# ---------- utils ----------

def window():
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

def to_date(x):
    return pd.to_datetime(x, errors="coerce")

def save_table(df: pd.DataFrame, path: Path, order: list[str]):
    df.loc[order].to_csv(path, encoding="utf-8")

def base_paths():
    # script is in extension/, so project root is one level above.
    scripts_dir = Path(__file__).resolve().parent   # .../extension
    project_root = scripts_dir.parent               # .../ (repo root)
    out = project_root / CFG["OUT_DIR"]
    out.mkdir(parents=True, exist_ok=True)
    return project_root, out


# ---------- loaders ----------

def load_similarity(project_root: Path) -> pd.DataFrame:
    p = project_root / CFG["SIM_FILE"]
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p} (run extension TF-IDF similarity first)")
    sim = pd.read_csv(p)
    sim["date"] = to_date(sim["date"])
    if "sim_tfidf" not in sim.columns:
        raise ValueError(f"Missing 'sim_tfidf' in {p}")
    return sim[["date", "sim_tfidf"]].dropna(subset=["date", "sim_tfidf"])

def load_controls(project_root: Path) -> pd.DataFrame:
    p = project_root / CFG["CTL_FILE"]
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p} (run replication controls step 7b first)")
    ctl = pd.read_csv(p)
    ctl["date_m"] = to_date(ctl["date_m"])
    return ctl

def load_car(project_root: Path) -> pd.DataFrame:
    p = project_root / CFG["CAR_FILE"]
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p} (run replication event study step 7 first)")
    car = pd.read_csv(p)
    car["date"] = to_date(car["date"])
    return car.dropna(subset=["date"])


# ---------- Table 3 ----------

def run_table3(sim: pd.DataFrame, ctl: pd.DataFrame, out: Path,
               sdt: pd.Timestamp, edt: pd.Timestamp, s: str, e: str):
    df = (sim[(sim["date"] >= sdt) & (sim["date"] <= edt)]
            .query("sim_tfidf > 0")
            .sort_values("date")
            .reset_index(drop=True))
    if len(df) < 5:
        raise ValueError(f"[Table3] Not enough obs in {s}..{e} (n={len(df)})")

    df["log_similarity"] = np.log(df["sim_tfidf"])
    df["time"] = np.log((df["date"] - sdt).dt.days + 1)
    df["time_count"] = np.log(np.arange(1, len(df) + 1))
    df["date_m"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.merge(ctl, on="date_m", how="left")

    controls = ["output_gap", "inflation", "delta_mro_eom"]

    specs = {
        "(1)": controls,
        "(2)": ["time"],
        "(3)": ["time"] + controls,
        "(4)": ["time_count"] + controls,
    }

    def col(m):
        return {
            "Intercept": fmt(m, "const"),
            "Time": fmt(m, "time"),
            "Time (count)": fmt(m, "time_count"),
            "Output gap": fmt(m, "output_gap"),
            "Inflation": fmt(m, "inflation"),
            "Delta MRO": fmt(m, "delta_mro_eom"),
            "Adjusted R2": f"{m.rsquared_adj*100:.2f}%",
        }

    table = pd.DataFrame({k: col(ols(df, "log_similarity", xs)) for k, xs in specs.items()})
    save_table(
        table,
        out / "table3_similarity_regressions_tfidf.csv",
        ["Intercept","Time","Time (count)","Output gap","Inflation","Delta MRO","Adjusted R2"]
    )

    keep = ["date","date_m","sim_tfidf","log_similarity","time","time_count"] + controls
    dfo = df[keep].copy()
    dfo["date"] = dfo["date"].dt.strftime("%Y-%m-%d")
    dfo["date_m"] = dfo["date_m"].dt.strftime("%Y-%m-%d")
    dfo.to_csv(out / "regression_dataset_table3_tfidf.csv", index=False, encoding="utf-8")

    print(f"[Table3 TF-IDF] Window: {s}->{e} | n={len(df)}")
    print(table.loc[["Intercept","Time","Time (count)","Output gap","Inflation","Delta MRO","Adjusted R2"]].to_string())


# ---------- Table 4 (log + z) ----------

def run_table4(car: pd.DataFrame, sim: pd.DataFrame, ctl: pd.DataFrame, out: Path,
               sdt: pd.Timestamp, edt: pd.Timestamp, s: str, e: str):
    controls = ["output_gap", "inflation", "delta_mro_eom"]
    y = "absCAR_pct"

    if y not in car.columns:
        raise ValueError(f"Missing '{y}' in CAR file (step 7 should create it).")
    if "pessimism_lm_pct" not in car.columns:
        raise ValueError("Missing 'pessimism_lm_pct' in CAR file (step 6 should create it).")

    car_w = (car[(car["date"] >= sdt) & (car["date"] <= edt)]
               .sort_values("date")
               .reset_index(drop=True))
    if len(car_w) == 0:
        raise ValueError(f"[Table4] No CAR rows in {s}..{e}")

    df = (car_w.assign(date_m=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
               .merge(ctl[["date_m"] + controls], on="date_m", how="left")
               .merge(sim, on="date", how="left"))

    # ---- (A) log(sim) spec (paper-like) ----
    dfl = df.query("sim_tfidf > 0").copy()
    if len(dfl) == 0:
        raise ValueError(f"[Table4 log] No observations with sim_tfidf > 0 in {s}..{e}")

    dfl["log_similarity"] = np.log(dfl["sim_tfidf"])
    dfl["pess_x_logsim"] = dfl["pessimism_lm_pct"] * dfl["log_similarity"]

    specs_log = {
        "(1)": ["pessimism_lm_pct"],
        "(2)": controls,
        "(3)": ["pessimism_lm_pct", "log_similarity", "pess_x_logsim"],
        "(4)": ["pessimism_lm_pct", "log_similarity", "pess_x_logsim"] + controls,
    }

    def col_log(m):
        return {
            "Intercept": fmt(m, "const"),
            "Pessimism": fmt(m, "pessimism_lm_pct"),
            "Similarity (log)": fmt(m, "log_similarity"),
            "Pessimism × similarity": fmt(m, "pess_x_logsim"),
            "Output gap": fmt(m, "output_gap"),
            "Inflation": fmt(m, "inflation"),
            "Delta MRO": fmt(m, "delta_mro_eom"),
            "Adjusted R²": f"{m.rsquared_adj*100:.2f}%",
        }

    # (1)(2) on df, (3)(4) on dfl
    m1 = ols(df,  y, specs_log["(1)"])
    m2 = ols(df,  y, specs_log["(2)"])
    m3 = ols(dfl, y, specs_log["(3)"])
    m4 = ols(dfl, y, specs_log["(4)"])

    table_log = pd.DataFrame({"(1)": col_log(m1), "(2)": col_log(m2), "(3)": col_log(m3), "(4)": col_log(m4)})
    save_table(
        table_log,
        out / "table4_absCAR_regressions_tfidf.csv",
        ["Intercept","Pessimism","Similarity (log)","Pessimism × similarity","Output gap","Inflation","Delta MRO","Adjusted R²"]
    )

    keep_log = ["date","date_m",y,"pessimism_lm_pct","sim_tfidf","log_similarity","pess_x_logsim"] + controls
    dfo = dfl[keep_log].copy()
    dfo["date"] = dfo["date"].dt.strftime("%Y-%m-%d")
    dfo["date_m"] = dfo["date_m"].dt.strftime("%Y-%m-%d")
    dfo.to_csv(out / "regression_dataset_table4_tfidf.csv", index=False, encoding="utf-8")

    print(f"[Table4 TF-IDF log] Window: {s}->{e} | n={len(df)} | n(sim>0)={len(dfl)}")
    print(table_log.loc[["Intercept","Pessimism","Similarity (log)","Pessimism × similarity","Output gap","Inflation","Delta MRO","Adjusted R²"]].to_string())

    # ---- (B) z(sim) spec (robustness) ----
    dfz = df[df["sim_tfidf"].notna()].copy()
    if len(dfz) == 0:
        print("[Table4 TF-IDF z] skipped: no sim_tfidf available")
        return

    ssim = dfz["sim_tfidf"].astype(float)
    sd = ssim.std(ddof=1)
    if not sd or np.isnan(sd) or sd <= 0:
        print("[Table4 TF-IDF z] skipped: sim_tfidf std is zero/NaN")
        return

    dfz["z_sim_tfidf"] = (ssim - ssim.mean()) / sd
    dfz["pess_x_zsim"] = dfz["pessimism_lm_pct"] * dfz["z_sim_tfidf"]

    specs_z = {
        "(3z)": ["pessimism_lm_pct", "z_sim_tfidf", "pess_x_zsim"],
        "(4z)": ["pessimism_lm_pct", "z_sim_tfidf", "pess_x_zsim"] + controls,
    }

    def col_z(m):
        return {
            "Intercept": fmt(m, "const"),
            "Pessimism": fmt(m, "pessimism_lm_pct"),
            "Similarity (z)": fmt(m, "z_sim_tfidf"),
            "Pessimism × similarity": fmt(m, "pess_x_zsim"),
            "Output gap": fmt(m, "output_gap"),
            "Inflation": fmt(m, "inflation"),
            "Delta MRO": fmt(m, "delta_mro_eom"),
            "Adjusted R²": f"{m.rsquared_adj*100:.2f}%",
        }

    mz3 = ols(dfz, y, specs_z["(3z)"])
    mz4 = ols(dfz, y, specs_z["(4z)"])

    table_z = pd.DataFrame({"(3z)": col_z(mz3), "(4z)": col_z(mz4)})
    save_table(
        table_z,
        out / "table4_absCAR_regressions_tfidf_z.csv",
        ["Intercept","Pessimism","Similarity (z)","Pessimism × similarity","Output gap","Inflation","Delta MRO","Adjusted R²"]
    )

    keep_z = ["date","date_m",y,"pessimism_lm_pct","sim_tfidf","z_sim_tfidf","pess_x_zsim"] + controls
    dfzo = dfz[keep_z].copy()
    dfzo["date"] = dfzo["date"].dt.strftime("%Y-%m-%d")
    dfzo["date_m"] = dfzo["date_m"].dt.strftime("%Y-%m-%d")
    dfzo.to_csv(out / "regression_dataset_table4_tfidf_z.csv", index=False, encoding="utf-8")

    print(f"[Table4 TF-IDF z] Window: {s}->{e} | n={len(dfz)}")
    print(table_z.loc[["Intercept","Pessimism","Similarity (z)","Pessimism × similarity","Output gap","Inflation","Delta MRO","Adjusted R²"]].to_string())


def main():
    project_root, out = base_paths()
    sdt, edt, s, e = window()

    sim = load_similarity(project_root)
    ctl = load_controls(project_root)
    car = load_car(project_root)

    run_table3(sim, ctl, out, sdt, edt, s, e)
    print("\n" + "-" * 80 + "\n")
    run_table4(car, sim, ctl, out, sdt, edt, s, e)


if __name__ == "__main__":
    main()
