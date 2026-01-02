# E8_merge_car_pessimism_similarity.py
# Merge event study output (CAR/absCAR + pessimism) with TF-IDF cosine similarity.

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

CONFIG = {
    "INPUT_CAR_PESS": "data_clean/ecb_pessimism_with_car.csv",
    "INPUT_TFIDF_SIM": "data_features/ecb_similarity_tfidf.csv",
    "OUTPUT_MERGED": "data_clean/ecb_analysis_dataset.csv",
    "EPS": 1e-6,

    # Optional filters to match the paper sample (set None to disable)
    "DATE_MIN": "1999-01-01",
    "DATE_MAX": "2013-12-31",
}


def main() -> None:
    # Script is in extension/, so project root is one level above.
    scripts_dir = Path(__file__).resolve().parent   # .../extension
    project_root = scripts_dir.parent               # .../ (repo root)

    p_car = project_root / CONFIG["INPUT_CAR_PESS"]
    p_sim = project_root / CONFIG["INPUT_TFIDF_SIM"]
    p_out = project_root / CONFIG["OUTPUT_MERGED"]
    p_out.parent.mkdir(parents=True, exist_ok=True)

    if not p_car.exists():
        raise FileNotFoundError(f"Missing: {p_car} (run replication step 7 first)")
    if not p_sim.exists():
        raise FileNotFoundError(f"Missing: {p_sim} (run extension TF-IDF similarity first)")

    car = pd.read_csv(p_car)
    sim = pd.read_csv(p_sim)

    if "date" not in car.columns:
        raise ValueError(f"Missing 'date' in {p_car}")
    if "date" not in sim.columns:
        raise ValueError(f"Missing 'date' in {p_sim}")

    # --- harmonize date formats ---
    car = car.copy()
    sim = sim.copy()
    car["date"] = pd.to_datetime(car["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sim["date"] = pd.to_datetime(sim["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    car = car[car["date"].notna()].copy()
    sim = sim[sim["date"].notna()].copy()

    # keep only needed cols from sim
    needed_sim_cols = ["date"]
    if "sim_tfidf" in sim.columns:
        needed_sim_cols += ["sim_tfidf"]
    if "log_sim_tfidf" in sim.columns:
        needed_sim_cols += ["log_sim_tfidf"]

    sim2 = sim[needed_sim_cols].copy()

    if "sim_tfidf" not in sim2.columns:
        raise ValueError("TF-IDF file must contain 'sim_tfidf' to merge.")

    # recompute log if not present
    if "log_sim_tfidf" not in sim2.columns:
        eps = float(CONFIG["EPS"])
        sim2["log_sim_tfidf"] = np.log(pd.to_numeric(sim2["sim_tfidf"], errors="coerce") + eps)

    # --- merge ---
    merged = car.merge(sim2, on="date", how="left")

    # Optional: filter to paper-like period
    merged["date_dt"] = pd.to_datetime(merged["date"], errors="coerce")
    if CONFIG["DATE_MIN"]:
        merged = merged[merged["date_dt"] >= pd.to_datetime(CONFIG["DATE_MIN"])]
    if CONFIG["DATE_MAX"]:
        merged = merged[merged["date_dt"] <= pd.to_datetime(CONFIG["DATE_MAX"])]

    # Add z-score versions (recommended for comparability)
    if merged["sim_tfidf"].notna().any():
        s = pd.to_numeric(merged["sim_tfidf"], errors="coerce")
        sd = s.std(ddof=1)
        if sd and not np.isnan(sd) and sd > 0:
            merged["z_sim_tfidf"] = (s - s.mean()) / sd
        else:
            merged["z_sim_tfidf"] = np.nan

    # Basic diagnostics (avoid KeyError if some columns don't exist)
    print("Rows:", len(merged))
    print("Missing sim_tfidf:", int(merged["sim_tfidf"].isna().sum()))
    if "absCAR" in merged.columns:
        print("Missing absCAR:", int(merged["absCAR"].isna().sum()))
    elif "absCAR_pct" in merged.columns:
        print("Missing absCAR_pct:", int(merged["absCAR_pct"].isna().sum()))

    merged.drop(columns=["date_dt"], inplace=True, errors="ignore")
    merged.to_csv(p_out, index=False, encoding="utf-8")
    print(f"Saved: {p_out}")


if __name__ == "__main__":
    main()