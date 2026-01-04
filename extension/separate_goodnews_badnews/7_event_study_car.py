# 7_event_study_car.py
# Compute CAR around ECB events using a constant-mean return model with fixed estimation and event windows.
# I/O:
#   Inputs : data_raw/^SX5E data.xlsx (prices),
#            data_clean/ecb_pessimism_lm.csv (or an alternative sentiment file if present).
#   Outputs: data_clean/ecb_event_study_car.csv (CAR table)
#            data_clean/ecb_pessimism_with_car.csv (merged dataset; keeps ALL sentiment columns).
#
# Key fix vs your failing run:
#   - Robust project-root detection (works from extension/... subfolders too)
#     so paths resolve to <root>/data_raw/... not <root>/extension/data_raw/...
#
# Notes:
#   For each ECB event date, align to a trading day, estimate mean return on [-250,-50],
#   and compute CAR on [-5,+5], then merge CAR back onto the event-level dataset.

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG = {
    "ALIGN_MARKET_DEFAULT": "next",  # "next" or "previous"
    "SX5E_XLSX": "^SX5E data.xlsx",
    # Prefer richer sentiment file if you created one (contains pess_neg_lm_pct / pess_pos_lm_pct).
    # The script will pick the FIRST one that exists.
    "EVENT_INPUT_CANDIDATES": [
        "ecb_sentiment_lm.csv",
        "ecb_pessimism_lm.csv",
    ],
    "OUT_CAR_TABLE": "ecb_event_study_car.csv",
    "OUT_MERGED": "ecb_pessimism_with_car.csv",
}


def get_project_root() -> Path:
    """
    Robustly find repo root even if this script is run from:
      <root>/replication/...
      <root>/extension/...
      <root>/extension/separate_goodnews_badnews/...
    We define root as the first parent containing BOTH data_raw/ and data_clean/.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_raw").exists() and (p / "data_clean").exists():
            return p
    raise RuntimeError("Could not locate project root (expected data_raw/ and data_clean/ in a parent folder).")


def resolve_paths(project_root: Path) -> tuple[Path, Path, Path]:
    """
    Return (sx5e_xlsx_path, event_input_csv_path, clean_dir_path).
    event_input_csv_path is selected from CONFIG['EVENT_INPUT_CANDIDATES'].
    """
    raw = project_root / "data_raw"
    clean = project_root / "data_clean"
    clean.mkdir(parents=True, exist_ok=True)

    sx5e = raw / CONFIG["SX5E_XLSX"]

    chosen = None
    for name in CONFIG["EVENT_INPUT_CANDIDATES"]:
        p = clean / name
        if p.exists():
            chosen = p
            break
    if chosen is None:
        # keep a helpful error message listing what we tried
        tried = ", ".join(str(clean / n) for n in CONFIG["EVENT_INPUT_CANDIDATES"])
        raise FileNotFoundError(f"Missing event-level input file. Tried: {tried}")

    return sx5e, chosen, clean


def load_sx5e(xlsx: Path) -> pd.DataFrame:
    """Load SX5E price data from Excel into clean OHLC with parsed dates."""
    df = pd.read_excel(xlsx)
    cols = {c.lower(): c for c in df.columns}
    for needed in ["date", "open", "high", "low", "close"]:
        if needed not in cols:
            raise ValueError(f"Missing column '{needed}' in {xlsx}. Found columns: {list(df.columns)}")

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
    """Compute log returns from Close prices and return ['Date','ret']."""
    df = df_px.copy()
    df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def load_events(event_path: Path) -> pd.DataFrame:
    """
    Load event-level dataset (pessimism/sentiment) and keep ALL columns.
    Deduplicate by date (keep first after sorting).
    """
    ecb = pd.read_csv(event_path)
    if "date" not in ecb.columns:
        raise ValueError(f"Event file is missing 'date' column: {event_path}")

    ecb["date"] = pd.to_datetime(ecb["date"], errors="coerce")
    ecb = (
        ecb.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="first")
        .reset_index(drop=True)
    )
    if len(ecb) == 0:
        raise ValueError(f"{event_path.name} is empty after parsing dates.")
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


def car_constant_mean(df_ret: pd.DataFrame, d: pd.Timestamp, align_mode: str) -> dict:
    """Compute CAR on [-5,+5] using mean returns from [-250,-50] around an aligned event trading date."""
    idx = pd.DatetimeIndex(df_ret["Date"])
    td = align_day(d, idx, align_mode)
    if pd.isna(td):
        return {"event_trading_date": pd.NaT, "CAR": np.nan, "absCAR": np.nan}

    i = idx.get_loc(td)
    est0, est1, evt0, evt1 = i - 250, i - 50, i - 5, i + 5
    if est0 < 0 or evt0 < 0 or evt1 >= len(df_ret):
        return {"event_trading_date": td, "CAR": np.nan, "absCAR": np.nan}

    mu = df_ret.iloc[est0 : est1 + 1]["ret"].mean()
    car = float((df_ret.iloc[evt0 : evt1 + 1]["ret"] - mu).sum())
    return {"event_trading_date": td, "CAR": car, "absCAR": abs(car)}


def compute_car_table(df_ret: pd.DataFrame, ecb: pd.DataFrame, align_mode: str) -> pd.DataFrame:
    """Compute CAR and absCAR (+ pct) for each event date and return CAR table."""
    rows = []
    for d in ecb["date"]:
        r = car_constant_mean(df_ret, d, align_mode)
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
    """Write data_clean/ecb_event_study_car.csv."""
    p = clean_dir / CONFIG["OUT_CAR_TABLE"]
    out_export = out.copy()
    for c in ["date", "event_trading_date"]:
        out_export[c] = pd.to_datetime(out_export[c], errors="coerce").dt.strftime("%Y-%m-%d")
    out_export.to_csv(p, index=False, encoding="utf-8")
    return p


def merge_with_events(ecb: pd.DataFrame, out: pd.DataFrame, clean_dir: Path) -> Path:
    """
    Merge CAR columns back onto the event-level dataset and write data_clean/ecb_pessimism_with_car.csv.
    IMPORTANT: keeps ALL original columns from the event dataset (including pess_neg_lm_pct / pess_pos_lm_pct if present).
    """
    p = clean_dir / CONFIG["OUT_MERGED"]

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
    sx5e_path, event_path, clean_dir = resolve_paths(project_root)

    align_mode = os.getenv("ECB_ALIGN_MARKET", CONFIG["ALIGN_MARKET_DEFAULT"]).strip().lower()
    if align_mode not in {"next", "previous"}:
        raise ValueError("ECB_ALIGN_MARKET must be 'next' or 'previous'.")

    if not sx5e_path.exists():
        raise FileNotFoundError(f"Missing market data file: {sx5e_path}")

    df_ret = compute_returns(load_sx5e(sx5e_path))
    ecb = load_events(event_path)

    print(f"Project root: {project_root}")
    print(f"Using event file: {event_path}")
    print(
        f"Events loaded: {len(ecb)} | {ecb['date'].min().date()} -> {ecb['date'].max().date()} | "
        f"ALIGN_MARKET={align_mode}"
    )

    out = compute_car_table(df_ret, ecb, align_mode)
    p_car = export_car_table(out, clean_dir)
    p_merged = merge_with_events(ecb, out, clean_dir)

    print(f"Saved: {p_car}")
    print(f"Saved merged: {p_merged}")

    # Helpful check for your asymmetry extension:
    need_cols = ["pess_neg_lm_pct", "pess_pos_lm_pct"]
    missing = [c for c in need_cols if c not in ecb.columns]
    if missing:
        print(
            "\n[Note] Your event file does NOT contain columns needed for good/bad news asymmetry: "
            + ", ".join(missing)
            + "\nMake sure your sentiment step outputs them into the chosen event file above "
            "(e.g., data_clean/ecb_sentiment_lm.csv), then rerun this script.\n"
        )


if __name__ == "__main__":
    main()
