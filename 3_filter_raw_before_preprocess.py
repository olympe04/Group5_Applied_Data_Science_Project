# 3_filter_raw_before_preprocess.py
# Filter BEFORE preprocessing (starting from data_raw/ecb_statements_raw.csv)
#
# Rules (to match the paper's 1999–2013 sample size of 172):
# 1) Drop specific speeches identified by (date + title) and/or by title (provided lists)
# 2) Drop specific speeches identified by subtitle (exact normalized match)
#
# Output:
# - data_raw/ecb_statements_raw_filtered.csv (or overwrite the original)
# - prints at the end the list of removed speeches (date, title, subtitle, url, reason)
# - also saves data_raw/ecb_statements_raw_removed.csv

from pathlib import Path
import re
import pandas as pd

# CONFIG
OVERWRITE_ORIGINAL = False
INPUT_PATH = "data_raw/ecb_statements_raw.csv"
OUTPUT_PATH = "data_raw/ecb_statements_raw_filtered.csv"

# A) Drop by (date, title)
DROP_BY_DATE_AND_TITLE = [
    {"date": "2000-03-30", "title": "ECB Press conference: Introductory statement"},
    {"date": "2000-10-19", "title": "ECB Press conference: Introductory statement"},
]

# B) Drop by title (regardless of date)
DROP_BY_TITLE = [
    "ECB Press Conference for the opening of the euro banknote design exhibition",
    "ECB press conference on the occasion of the Signing of the Agreement between the European Central Bank and Europol",
    "Euro area balance of payments and international investment position vis-à-vis main external counterparts",
    "Introductory statement on the winning design chosen in the international urban planning and architectural design competition for the new ECB premises",
    "Introductory statement to the press conference on the chosen design of the international urban planning and architectural design competition for the New ECB Premises",
    "Joint press conference given by the EU Presidency, the European Central Bank, and the European Commission on the introduction of the euro banknotes and coins, Frankfurt am Main, 3 January 2002",
    "Transcript of the Press Briefing",  # just a brief speech before a Q&A
    "Press seminar on the evaluation of the ECB's monetary policy strategy",  # essentially empty (only a PDF + Q&A)
    "Transcript of the comprehensive assessment press conference (with Q&A)",
]

# C) Drop by subtitle (exact normalized match)
DROP_BY_SUBTITLE = [
    'Dr. Willem F. Duisenberg, President of the European Central Bank, on the occasion of the signing of the TACIS "Central Bank Training" contract and of a Protocol between the European Central Bank, the Delegation of the European Commission in Russia and the Central Bank of Russia, Moscow, 13 October 2003.'
]


# ----------------
# Helpers
# ----------------
def norm_text(x: str) -> str:
    """Robust normalization (whitespace + casing) for matching text fields."""
    return re.sub(r"\s+", " ", "" if x is None else str(x)).strip().lower()


def main():
    base_dir = Path(__file__).resolve().parent
    in_path = base_dir / INPUT_PATH
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path} (run step 02 first)")

    df = pd.read_csv(in_path)
    if not {"date", "title"}.issubset(df.columns):
        raise ValueError(f"Missing required columns in {in_path}. Need at least: ['date', 'title']")

    # Normalize fields used for matching
    df = df.copy()
    df["date"] = df["date"].fillna("").astype(str).str.slice(0, 10)
    df["title_norm"] = df["title"].map(norm_text)
    df["subtitle_norm"] = df["subtitle"].map(norm_text) if "subtitle" in df.columns else ""

    removed = []

    # 1) Drop by (date, title)
    pairs = {(d["date"].strip(), norm_text(d["title"])) for d in DROP_BY_DATE_AND_TITLE if d.get("date") and d.get("title")}
    if pairs:
        m = df.apply(lambda r: (r["date"], r["title_norm"]) in pairs, axis=1)
        if m.any():
            x = df.loc[m].copy()
            x["drop_reason"] = "drop_by_date_and_title"
            removed.append(x)
        df = df.loc[~m].copy()

    # 2) Drop by title
    titles = {norm_text(t) for t in DROP_BY_TITLE if str(t).strip()}
    if titles:
        m = df["title_norm"].isin(titles)
        if m.any():
            x = df.loc[m].copy()
            x["drop_reason"] = "drop_by_title"
            removed.append(x)
        df = df.loc[~m].copy()

    # 3) Drop by subtitle
    subs = {norm_text(s) for s in DROP_BY_SUBTITLE if str(s).strip()}
    if subs:
        m = df["subtitle_norm"].isin(subs)
        if m.any():
            x = df.loc[m].copy()
            x["drop_reason"] = "drop_by_subtitle"
            removed.append(x)
        df = df.loc[~m].copy()

    print(f"Remaining rows: {len(df)}")

    # Write filtered output
    out_path = in_path if OVERWRITE_ORIGINAL else (base_dir / OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in ["title_norm", "subtitle_norm"] if c in df.columns]).to_csv(
        out_path, index=False, encoding="utf-8"
    )
    print(f"Saved: {out_path}")

    # Report removed speeches + save CSV
    if removed:
        rem = pd.concat(removed, ignore_index=True)
        report_cols = [c for c in ["date", "title", "subtitle", "url", "drop_reason"] if c in rem.columns]
        rem = rem[report_cols].drop_duplicates()
        if {"date", "drop_reason", "title"}.issubset(rem.columns):
            rem = rem.sort_values(["date", "drop_reason", "title"], kind="mergesort")

        print("\n REMOVED SPEECHES (date, title, subtitle, url, reason)")
        pd.set_option("display.max_colwidth", 180)
        pd.set_option("display.width", 180)
        pd.set_option("display.max_rows", 5000)
        print(rem.to_string(index=False))

        removed_csv = out_path.parent / "ecb_statements_raw_removed.csv"
        rem.to_csv(removed_csv, index=False, encoding="utf-8")
        print(f"\nSaved removed list: {removed_csv}")
    else:
        print("\nNo rows removed by the configured rules.")


if __name__ == "__main__":
    main()
