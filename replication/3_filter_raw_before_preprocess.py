# 3_filter_raw_before_preprocess.py
# Filter raw ECB transcript rows before preprocessing by removing known non-target/special documents.
# I/O:
#   Inputs: CSV file "data_raw/ecb_statements_raw.csv".
#   Outputs: CSV "data_raw/ecb_statements_raw_filtered.csv" (kept rows) and "data_raw/ecb_statements_raw_removed.csv" (dropped rows report).

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


CONFIG = {
    "OVERWRITE_ORIGINAL": False,
    "INPUT_PATH": "data_raw/ecb_statements_raw.csv",
    "OUTPUT_PATH": "data_raw/ecb_statements_raw_filtered.csv",
}


DROP_BY_DATE_AND_TITLE = [
    {"date": "2000-03-30", "title": "ECB Press conference: Introductory statement"},
    {"date": "2000-10-19", "title": "ECB Press conference: Introductory statement"},
]

DROP_BY_TITLE = [
    "ECB Press Conference for the opening of the euro banknote design exhibition",
    "ECB press conference on the occasion of the Signing of the Agreement between the European Central Bank and Europol",
    "Euro area balance of payments and international investment position vis-à-vis main external counterparts",
    "Introductory statement on the winning design chosen in the international urban planning and architectural design competition for the new ECB premises",
    "Introductory statement to the press conference on the chosen design of the international urban planning and architectural design competition for the New ECB Premises",
    "Joint press conference given by the EU Presidency, the European Central Bank, and the European Commission on the introduction of the euro banknotes and coins, Frankfurt am Main, 3 January 2002",
    "Transcript of the Press Briefing",
    "Press seminar on the evaluation of the ECB's monetary policy strategy",
    "Transcript of the comprehensive assessment press conference (with Q&A)",
]

DROP_BY_SUBTITLE = [
    'Dr. Willem F. Duisenberg, President of the European Central Bank, on the occasion of the signing of the TACIS "Central Bank Training" contract and of a Protocol between the European Central Bank, the Delegation of the European Commission in Russia and the Central Bank of Russia, Moscow, 13 October 2003.'
]


def norm_text(x: str) -> str:
    """Lowercase and normalize whitespace for robust, case-insensitive string matching."""
    return re.sub(r"\s+", " ", "" if x is None else str(x)).strip().lower()


def apply_drops(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows matching configured drop rules and return both kept and removed datasets."""
    df = df.copy()

    df["date"] = df["date"].fillna("").astype(str).str.slice(0, 10)
    df["title_norm"] = df["title"].map(norm_text)
    df["subtitle_norm"] = df["subtitle"].map(norm_text) if "subtitle" in df.columns else ""

    removed_chunks: List[pd.DataFrame] = []

    # 1) Drop by (date, title)
    pairs = {
        (d["date"].strip(), norm_text(d["title"]))
        for d in DROP_BY_DATE_AND_TITLE
        if d.get("date") and d.get("title")
    }
    if pairs:
        m = df.apply(lambda r: (r["date"], r["title_norm"]) in pairs, axis=1)
        if m.any():
            x = df.loc[m].copy()
            x["drop_reason"] = "drop_by_date_and_title"
            removed_chunks.append(x)
        df = df.loc[~m].copy()

    # 2) Drop by title
    titles = {norm_text(t) for t in DROP_BY_TITLE if str(t).strip()}
    if titles:
        m = df["title_norm"].isin(titles)
        if m.any():
            x = df.loc[m].copy()
            x["drop_reason"] = "drop_by_title"
            removed_chunks.append(x)
        df = df.loc[~m].copy()

    # 3) Drop by subtitle
    subs = {norm_text(s) for s in DROP_BY_SUBTITLE if str(s).strip()}
    if subs:
        m = df["subtitle_norm"].isin(subs)
        if m.any():
            x = df.loc[m].copy()
            x["drop_reason"] = "drop_by_subtitle"
            removed_chunks.append(x)
        df = df.loc[~m].copy()

    removed = pd.concat(removed_chunks, ignore_index=True) if removed_chunks else pd.DataFrame()
    return df, removed


def main() -> None:
    """Load raw transcripts, apply drop rules, and write kept rows plus a removal report to disk."""
    scripts_dir = Path(__file__).resolve().parent   # .../replication
    project_root = scripts_dir.parent               # .../ (repo root)

    in_path = project_root / CONFIG["INPUT_PATH"]
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path} (run scraping step first)")

    df = pd.read_csv(in_path)
    if not {"date", "title"}.issubset(df.columns):
        raise ValueError(f"Missing required columns in {in_path}. Need at least: ['date', 'title']")

    df_kept, removed = apply_drops(df)

    print(f"Remaining rows after drops: {len(df_kept)}")

    out_path = in_path if CONFIG["OVERWRITE_ORIGINAL"] else (project_root / CONFIG["OUTPUT_PATH"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_kept.drop(columns=["title_norm", "subtitle_norm"], errors="ignore").to_csv(
        out_path, index=False, encoding="utf-8"
    )
    print(f"Saved: {out_path}")

    removed_csv = out_path.parent / "ecb_statements_raw_removed.csv"
    if len(removed):
        report_cols = [c for c in ["date", "title", "subtitle", "url", "drop_reason"] if c in removed.columns]
        rep = removed[report_cols].drop_duplicates()
        if {"date", "drop_reason", "title"}.issubset(rep.columns):
            rep = rep.sort_values(["date", "drop_reason", "title"], kind="mergesort")
        rep.to_csv(removed_csv, index=False, encoding="utf-8")
        print(f"Saved removed list: {removed_csv} ({len(rep)} rows)")
    else:
        pd.DataFrame(columns=["date", "title", "subtitle", "url", "drop_reason"]).to_csv(
            removed_csv, index=False, encoding="utf-8"
        )
        print(f"Saved removed list: {removed_csv} (0 rows)")


if __name__ == "__main__":
    main()
