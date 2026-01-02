# 4_pre-process.py
# Preprocess ECB statements into two parallel text representations (clean tokens and Porter stems).
# I/O:
#   Inputs: CSV file "data_raw/ecb_statements_raw_filtered.csv" (expects [url], optional [date], and text columns).
#   Outputs: CSV file "data_clean/ecb_statements_preprocessed.csv" with metadata plus [tokens_clean_str, stems_str, n_tokens_clean, n_stems].

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

CONFIG = {
    "INPUT_CSV": "data_raw/ecb_statements_raw_filtered.csv",
    "OUTPUT_CSV": "data_clean/ecb_statements_preprocessed.csv",
    "TEXT_SOURCE": "statement",
    "MIN_TOKEN_LEN": 2,
}


def _get_stopwords():
    """Load English stopwords from NLTK."""
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english")), "nltk"


def _get_stemmer():
    """Load an NLTK PorterStemmer."""
    return PorterStemmer(), "nltk_porter"


TOKEN_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*")
WHITESPACE_RE = re.compile(r"\s+")
NONBREAKING_RE = re.compile(r"\xa0")


def clean_text_basic(text: str) -> str:
    """Apply light normalization (spaces, apostrophes, dashes) to stabilize downstream tokenization."""
    if text is None:
        return ""
    t = str(text)
    t = NONBREAKING_RE.sub(" ", t)
    t = t.replace("\u2019", "'")
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text: str):
    """Lowercase and extract alphabetic tokens, allowing internal hyphens/apostrophes."""
    return TOKEN_RE.findall(clean_text_basic(text).lower())


def remove_stopwords(tokens, stopwords_set: set, min_len: int):
    """Filter out stopwords and short tokens based on a configurable minimum length."""
    return [w for w in tokens if len(w) >= min_len and w not in stopwords_set]


def preprocess_two_versions(text: str, stopwords_set: set, stemmer, min_len: int) -> dict:
    """Generate cleaned tokens (no stemming) and stemmed tokens (Porter) plus token counts."""
    tokens_raw = tokenize(text)
    tokens_clean = remove_stopwords(tokens_raw, stopwords_set=stopwords_set, min_len=min_len)
    stems = [stemmer.stem(w) for w in tokens_clean]
    return {
        "tokens_clean_str": " ".join(tokens_clean),
        "stems_str": " ".join(stems),
        "n_tokens_clean": len(tokens_clean),
        "n_stems": len(stems),
    }


def load_date_map(project_root: Path) -> pd.DataFrame:
    """Load an optional URL→date mapping from step 1 to fill missing dates in the main dataset."""
    p = project_root / "data_raw" / "ecb_speech_urls.csv"
    if not p.exists():
        return pd.DataFrame(columns=["url", "date_map"])

    dm = pd.read_csv(p)
    if not {"url", "date"}.issubset(dm.columns):
        return pd.DataFrame(columns=["url", "date_map"])

    out = dm[["url", "date"]].copy()
    out["url"] = out["url"].astype(str).str.strip()
    out["date_map"] = out["date"].astype(str).str.slice(0, 10)
    return out[["url", "date_map"]]


def fill_date_string(df: pd.DataFrame, date_map: pd.DataFrame) -> pd.DataFrame:
    """Ensure a YYYY-MM-DD 'date' column exists and fill missing values using the URL→date mapping when available."""
    df = df.copy()
    df["url"] = df["url"].fillna("").astype(str).str.strip()

    if "date" not in df.columns:
        df["date"] = ""
    df["date"] = df["date"].fillna("").astype(str).str.slice(0, 10)

    parsed = pd.to_datetime(df["date"], errors="coerce")
    print(f"Initial date parse: non-NaT = {parsed.notna().sum()} / {len(df)}")

    if len(date_map) == 0:
        return df

    df = df.merge(date_map, on="url", how="left")

    need_fill = parsed.isna()
    df.loc[need_fill, "date"] = df.loc[need_fill, "date_map"].fillna("").astype(str).str.slice(0, 10)

    parsed2 = pd.to_datetime(df["date"], errors="coerce")
    print(f"After fill:        non-NaT = {parsed2.notna().sum()} / {len(df)}")

    return df.drop(columns=["date_map"], errors="ignore")


def select_text_column(df: pd.DataFrame) -> pd.Series:
    """Select the configured text source column ('statement_text' or 'qa_text') for preprocessing."""
    src = str(CONFIG["TEXT_SOURCE"]).strip().lower()
    if src == "statement":
        return df.get("statement_text", "").fillna("").astype(str)
    if src == "qa":
        return df.get("qa_text", "").fillna("").astype(str)
    raise ValueError("CONFIG['TEXT_SOURCE'] must be 'statement' or 'qa'.")


def main() -> None:
    """Load filtered transcripts, preprocess the chosen text field, and write the enriched dataset to CSV."""
    scripts_dir = Path(__file__).resolve().parent   # .../replication
    project_root = scripts_dir.parent               # .../ (repo root)

    in_path = project_root / CONFIG["INPUT_CSV"]
    out_path = project_root / CONFIG["OUTPUT_CSV"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input introuvable: {in_path}")

    df = pd.read_csv(in_path)

    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.len().eq(0)].copy()

    if "url" not in df.columns:
        raise ValueError("Input missing 'url' column")

    date_map = load_date_map(project_root)
    df = fill_date_string(df, date_map)

    df["text_for_preprocess"] = select_text_column(df)
    df = df[df["text_for_preprocess"].str.strip().str.len() > 0].copy()
    print(f"Loaded rows (non-empty {CONFIG['TEXT_SOURCE']}): {len(df)}")

    stopwords_set, sw_src = _get_stopwords()
    stemmer, stem_src = _get_stemmer()
    print(f"Stopwords: {sw_src} | Stemmer: {stem_src}")

    out = df["text_for_preprocess"].apply(
        lambda x: preprocess_two_versions(x, stopwords_set, stemmer, CONFIG["MIN_TOKEN_LEN"])
    )

    df["tokens_clean_str"] = out.apply(lambda d: d["tokens_clean_str"])
    df["stems_str"] = out.apply(lambda d: d["stems_str"])
    df["n_tokens_clean"] = out.apply(lambda d: d["n_tokens_clean"])
    df["n_stems"] = out.apply(lambda d: d["n_stems"])

    keep_meta = [c for c in ["date", "url", "title", "subtitle", "method"] if c in df.columns]
    raw_col = "statement_text" if str(CONFIG["TEXT_SOURCE"]).strip().lower() == "statement" else "qa_text"

    keep_cols = keep_meta + [
        raw_col,
        "n_tokens_clean", "n_stems",
        "tokens_clean_str", "stems_str",
    ]

    df[keep_cols].to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved CSV: {out_path}")

    tmp = df[keep_cols].copy()
    tmp["date_dt"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp[tmp["date_dt"].notna()].sort_values("date_dt").head(3)

    if len(tmp):
        print("Sample (date, n_tokens_clean, first 12 stems):")
        for _, row in tmp.iterrows():
            stems_preview = " ".join(str(row["stems_str"]).split()[:12])
            print(row["date"], int(row["n_tokens_clean"]), stems_preview)
    else:
        print("Sample: no valid dates available to preview ordering.")


if __name__ == "__main__":
    main()
