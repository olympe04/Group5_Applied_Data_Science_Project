# E7_event_study_car_uncertainty.py
# Compute CAR around ECB events using a constant-mean return model with fixed estimation and event windows.
# I/O:
#   Inputs : data_raw/^SX5E data.xlsx (prices), data_clean/ecb_uncertainty_lm.csv (event dates).
#   Outputs: data_features/ecb_event_study_car_uncertainty.csv (CAR table)
#            data_features/ecb_uncertainty_with_car.csv (merged dataset).

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ALIGN_MARKET = "next"  # "next" or "previous"


def get_project_root() -> Path:
    """
    Robustly find project root even if this script is moved into subfolders like:
      <root>/extension/uncertainty/E7_event_study_car_uncertainty.py

    Strategy: walk up parents until we find expected root markers.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError(
        "Could not locate project root. Expected to find 'data_clean/' and 'outputs/' in a parent directory."
    )


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path]:
    raw = project_root / "data_raw"
    clean = project_root / "data_clean"            # input events still here
    features = project_root / "data_features"      # outputs go here
    features.mkdir(parents=True, exist_ok=True)
    return raw / "^SX5E data.xlsx", clean / "ecb_uncertainty_lm.csv", features


def load_sx5e(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx)
    cols = {c.lower(): c for c in df.columns}
    for needed in ["date", "open", "high", "low", "close"]:
        if needed not in cols:
            raise ValueError(f"Missing column '{needed}' in {xlsx}. Found: {list(df.columns)}")

    df = df[[cols["date"], cols["open"], cols["high"], cols["low"], cols["close"]]].copy()
    df.columns = ["Date", "Open", "High", "Low", "Close"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return (
        df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def compute_returns(df_px: pd.DataFrame) -> pd.DataFrame:
    df = df_px.copy()
    df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def load_events(unc_path: Path) -> pd.DataFrame:
    ecb = pd.read_csv(unc_path)
    if "date" not in ecb.columns:
        raise ValueError(f"Missing 'date' column in {unc_path}")

    ecb["date"] = pd.to_datetime(ecb["date"], errors="coerce")
    ecb = (
        ecb.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="first")
        .reset_index(drop=True)
    )
    if len(ecb) == 0:
        raise ValueError("ecb_uncertainty_lm.csv is empty after parsing dates.")
    return ecb


def align_day(d: pd.Timestamp, idx: pd.DatetimeIndex, mode: str) -> pd.Timestamp:
    if d in idx:
        return d
    if mode == "next":
        i = idx.searchsorted(d, side="left")
        return idx[i] if i < len(idx) else pd.NaT
    i = idx.searchsorted(d, side="right") - 1
    return idx[i] if i >= 0 else pd.NaT


def car_constant_mean(df_ret: pd.DataFrame, d: pd.Timestamp) -> dict:
    idx = pd.DatetimeIndex(df_ret["Date"])
    td = align_day(d, idx, ALIGN_MARKET)
    if pd.isna(td):
        return {"event_trading_date": pd.NaT, "CAR": np.nan, "absCAR": np.nan}

    i = idx.get_loc(td)
    est0, est1, evt0, evt1 = i - 250, i - 50, i - 5, i + 5
    if est0 < 0 or evt0 < 0 or evt1 >= len(df_ret):
        return {"event_trading_date": td, "CAR": np.nan, "absCAR": np.nan}

    mu = df_ret.iloc[est0 : est1 + 1]["ret"].mean()
    car = float((df_ret.iloc[evt0 : evt1 + 1]["ret"] - mu).sum())
    return {"event_trading_date": td, "CAR": car, "absCAR": abs(car)}


def compute_car_table(df_ret: pd.DataFrame, ecb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in ecb["date"]:
        r = car_constant_mean(df_ret, d)
        rows.append(
            {"date": d, "event_trading_date": r["event_trading_date"], "CAR": r["CAR"], "absCAR": r["absCAR"]}
        )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["CAR_pct"] = out["CAR"] * 100.0
    out["absCAR_pct"] = out["absCAR"] * 100.0
    return out


def export_car_table(out: pd.DataFrame, features_dir: Path) -> Path:
    p = features_dir / "ecb_event_study_car_uncertainty.csv"   # <- changed
    out_export = out.copy()
    for c in ["date", "event_trading_date"]:
        out_export[c] = pd.to_datetime(out_export[c], errors="coerce").dt.strftime("%Y-%m-%d")
    out_export.to_csv(p, index=False, encoding="utf-8")
    return p


def merge_with_uncertainty(ecb: pd.DataFrame, out: pd.DataFrame, features_dir: Path) -> Path:
    p = features_dir / "ecb_uncertainty_with_car.csv"          # <- changed

    ecb2 = ecb.copy()
    ecb2["date"] = pd.to_datetime(ecb2["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    out2 = out.copy()
    out2["date"] = pd.to_datetime(out2["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out2["event_trading_date"] = pd.to_datetime(out2["event_trading_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    merged = ecb2.merge(
        out2[["date", "event_trading_date", "CAR", "absCAR", "CAR_pct", "absCAR_pct"]],
        on="date",
        how="left",
    )
    merged.to_csv(p, index=False, encoding="utf-8")
    return p


def main() -> None:
    project_root = get_project_root()
    sx5e_path, unc_path, features_dir = resolve_paths(project_root)

    if not sx5e_path.exists():
        raise FileNotFoundError(f"Missing market data file: {sx5e_path}")
    if not unc_path.exists():
        raise FileNotFoundError(f"Missing uncertainty file: {unc_path} (run E6_uncertainty_lm.py first)")

    df_ret = compute_returns(load_sx5e(sx5e_path))
    ecb = load_events(unc_path)

    print(
        f"Events loaded for event study: {len(ecb)} | "
        f"{ecb['date'].min().date()} -> {ecb['date'].max().date()}"
    )

    out = compute_car_table(df_ret, ecb)
    p_car = export_car_table(out, features_dir)
    p_merged = merge_with_uncertainty(ecb, out, features_dir)

    print(f"Saved: {p_car}")
    print(f"Saved merged: {p_merged}")


if __name__ == "__main__":
    main()
