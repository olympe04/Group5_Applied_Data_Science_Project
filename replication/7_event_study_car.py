# 7_event_study_car.py
# Compute CAR around ECB events using a constant-mean return model with fixed estimation and event windows.
# I/O:
#   Inputs : data_raw/^SX5E data.xlsx (prices), data_clean/ecb_pessimism_lm.csv (event dates).
#   Outputs: data_clean/ecb_event_study_car.csv (CAR table) and data_clean/ecb_pessimism_with_car.csv (merged dataset).
# Notes:
#   For each ECB event date, the script aligns to a trading day, estimates the mean return on [-250,-50],
#   and computes CAR on [-5,+5], then merges CAR back onto the event-level pessimism dataset.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ALIGN_MARKET = "next"  # "next" or "previous"


def get_project_root() -> Path:
    """Return repository root (script is in replication/)."""
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir.parent


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path]:
    """Return (sx5e_xlsx_path, pessimism_csv_path, clean_dir_path)."""
    raw = project_root / "data_raw"
    clean = project_root / "data_clean"
    clean.mkdir(parents=True, exist_ok=True)
    return raw / "^SX5E data.xlsx", clean / "ecb_pessimism_lm.csv", clean


def load_sx5e(xlsx: Path) -> pd.DataFrame:
    """Load SX5E price data from Excel into a clean OHLC dataframe with parsed dates."""
    df = pd.read_excel(xlsx)
    cols = {c.lower(): c for c in df.columns}
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
    """Compute log returns from Close prices and return a dataframe with ['Date','ret']."""
    df = df_px.copy()
    df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def load_events(pess_path: Path) -> pd.DataFrame:
    """Load event dates from pessimism CSV and return a date-sorted, deduplicated dataframe."""
    ecb = pd.read_csv(pess_path)
    ecb["date"] = pd.to_datetime(ecb["date"], errors="coerce")
    ecb = (
        ecb.dropna(subset=["date"])
           .sort_values("date")
           .drop_duplicates("date", keep="first")
           .reset_index(drop=True)
    )
    if len(ecb) == 0:
        raise ValueError("ecb_pessimism_lm.csv is empty after parsing dates.")
    return ecb


def align_day(d: pd.Timestamp, idx: pd.DatetimeIndex, mode: str) -> pd.Timestamp:
    """Align a date to the closest trading day using 'next' or 'previous' within the given index."""
    if d in idx:
        return d
    if mode == "next":
        i = idx.searchsorted(d, side="left")
        return idx[i] if i < len(idx) else pd.NaT
    i = idx.searchsorted(d, side="right") - 1
    return idx[i] if i >= 0 else pd.NaT


def car_constant_mean(df_ret: pd.DataFrame, d: pd.Timestamp) -> dict:
    """Compute CAR on [-5,+5] using mean returns from [-250,-50] around an aligned event trading date."""
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
    """Compute CAR and absCAR (plus % versions) for each event date and return the CAR table."""
    rows = []
    for d in ecb["date"]:
        r = car_constant_mean(df_ret, d)
        rows.append(
            {
                "date": d,
                "event_trading_date": r["event_trading_date"],
                "CAR": r["CAR"],
                "absCAR": r["absCAR"],
            }
        )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["CAR_pct"] = out["CAR"] * 100.0
    out["absCAR_pct"] = out["absCAR"] * 100.0
    return out


def export_car_table(out: pd.DataFrame, clean_dir: Path) -> Path:
    """Export the CAR table to data_clean/ecb_event_study_car.csv and return the written path."""
    p = clean_dir / "ecb_event_study_car.csv"
    out_export = out.copy()
    for c in ["date", "event_trading_date"]:
        out_export[c] = pd.to_datetime(out_export[c], errors="coerce").dt.strftime("%Y-%m-%d")
    out_export.to_csv(p, index=False, encoding="utf-8")
    return p


def merge_with_pessimism(ecb: pd.DataFrame, out: pd.DataFrame, clean_dir: Path) -> Path:
    """Merge CAR columns back onto the pessimism dataset and write data_clean/ecb_pessimism_with_car.csv."""
    p = clean_dir / "ecb_pessimism_with_car.csv"

    ecb2 = ecb.copy()
    ecb2["date"] = ecb2["date"].dt.strftime("%Y-%m-%d")

    out2 = out.copy()
    out2["date"] = out2["date"].dt.strftime("%Y-%m-%d")
    out2["event_trading_date"] = pd.to_datetime(out2["event_trading_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    merged = ecb2.merge(
        out2[["date", "event_trading_date", "CAR", "absCAR", "CAR_pct", "absCAR_pct"]],
        on="date",
        how="left",
    )
    merged.to_csv(p, index=False, encoding="utf-8")
    return p


def main() -> None:
    """Execute the event study: load inputs, compute CAR, export CAR table, and export merged dataset."""
    project_root = get_project_root()
    sx5e_path, pess_path, clean_dir = resolve_paths(project_root)

    if not sx5e_path.exists():
        raise FileNotFoundError(f"Missing market data file: {sx5e_path}")
    if not pess_path.exists():
        raise FileNotFoundError(f"Missing pessimism file: {pess_path} (run step 6 first)")

    df_ret = compute_returns(load_sx5e(sx5e_path))
    ecb = load_events(pess_path)

    print(
        f"Events loaded for event study: {len(ecb)} | "
        f"{ecb['date'].min().date()} -> {ecb['date'].max().date()}"
    )

    out = compute_car_table(df_ret, ecb)
    p_car = export_car_table(out, clean_dir)
    p_merged = merge_with_pessimism(ecb, out, clean_dir)

    print(f"Saved: {p_car}")
    print(f"Saved merged: {p_merged}")


if __name__ == "__main__":
    main()