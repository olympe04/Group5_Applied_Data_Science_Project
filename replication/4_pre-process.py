# 4_pre-process.py
# Preprocess ECB statements into two parallel text representations (clean tokens and Porter stems).
# I/O:
#   Inputs : data_raw/ecb_statements_raw_filtered.csv (expects [url], optional [date], and text columns).
#   Outputs: data_clean/ecb_statements_preprocessed.csv with metadata plus
#            [tokens_clean_str, stems_str, n_tokens_clean, n_stems].
# Notes:
#   The script selects either the statement or Q&A text, normalizes + tokenizes, removes stopwords,
#   computes Porter stems, and writes the enriched dataset for downstream similarity/sentiment steps.

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
    "TEXT_SOURCE": "statement",  # "statement" or "qa"
    "MIN_TOKEN_LEN": 2,
}


TOKEN_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*")
WHITESPACE_RE = re.compile(r"\s+")
NONBREAKING_RE = re.compile(r"\xa0")


def get_project_root() -> Path:
    """Return repository root (script is in replication/)."""
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir.parent


def resolve_paths(project_root: Path) -> tuple[Path, Path]:
    """Return (input_csv_path, output_csv_path) for preprocessing."""
    in_path = project_root / CONFIG["INPUT_CSV"]
    out_path = project_root / CONFIG["OUTPUT_CSV"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return in_path, out_path


def get_stopwords() -> tuple[set[str], str]:
    """Return English stopwords set (NLTK) and a source label."""
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english")), "nltk"


def get_stemmer() -> tuple[PorterStemmer, str]:
    """Return Porter stemmer instance (NLTK) and a source label."""
    return PorterStemmer(), "nltk_porter"


def clean_text_basic(text: str) -> str:
    """Normalize spaces, apostrophes, and dashes to stabilize tokenization."""
    if text is None:
        return ""
    t = str(text)
    t = NONBREAKING_RE.sub(" ", t)
    t = t.replace("\u2019", "'")
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    return WHITESPACE_RE.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphabetic tokens allowing internal hyphens/apostrophes."""
    return TOKEN_RE.findall(clean_text_basic(text).lower())


def remove_stopwords(tokens: list[str], stopwords_set: set[str], min_len: int) -> list[str]:
    """Filter stopwords and short tokens according to the minimum length."""
    return [w for w in tokens if len(w) >= min_len and w not in stopwords_set]


def preprocess_two_versions(text: str, stopwords_set: set[str], stemmer: PorterStemmer, min_len: int) -> dict:
    """Return cleaned tokens + Porter stems strings and their counts for one document."""
    tokens_clean = remove_stopwords(tokenize(text), stopwords_set=stopwords_set, min_len=min_len)
    stems = [stemmer.stem(w) for w in tokens_clean]
    return {
        "tokens_clean_str": " ".join(tokens_clean),
        "stems_str": " ".join(stems),
        "n_tokens_clean": len(tokens_clean),
        "n_stems": len(stems),
    }


def load_date_map(project_root: Path) -> pd.DataFrame:
    """Return URL→date mapping from data_raw/ecb_speech_urls.csv (empty if unavailable)."""
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
    """Ensure a YYYY-MM-DD 'date' exists and fill missing dates using URL→date mapping when available."""
    d = df.copy()
    d["url"] = d["url"].fillna("").astype(str).str.strip()
    d["date"] = d.get("date", "").fillna("").astype(str).str.slice(0, 10)

    parsed = pd.to_datetime(d["date"], errors="coerce")
    print(f"Initial date parse: non-NaT = {parsed.notna().sum()} / {len(d)}")

    if len(date_map):
        d = d.merge(date_map, on="url", how="left")
        d.loc[parsed.isna(), "date"] = d.loc[parsed.isna(), "date_map"].fillna("").astype(str).str.slice(0, 10)
        parsed2 = pd.to_datetime(d["date"], errors="coerce")
        print(f"After fill:        non-NaT = {parsed2.notna().sum()} / {len(d)}")
        d = d.drop(columns=["date_map"], errors="ignore")

    return d


def select_text_column(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return (text_series, raw_text_column_name) according to CONFIG['TEXT_SOURCE']."""
    src = str(CONFIG["TEXT_SOURCE"]).strip().lower()
    if src == "statement":
        return df.get("statement_text", "").fillna("").astype(str), "statement_text"
    if src == "qa":
        return df.get("qa_text", "").fillna("").astype(str), "qa_text"
    raise ValueError("CONFIG['TEXT_SOURCE'] must be 'statement' or 'qa'.")


def load_input(in_path: Path) -> pd.DataFrame:
    """Load filtered transcripts and drop rows with an 'error' flag if present."""
    if not in_path.exists():
        raise FileNotFoundError(f"Input introuvable: {in_path}")
    df = pd.read_csv(in_path)
    if "url" not in df.columns:
        raise ValueError("Input missing 'url' column")
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.len().eq(0)].copy()
    return df


def add_preprocessed_columns(
    df: pd.DataFrame, stopwords_set: set[str], stemmer: PorterStemmer, min_len: int
) -> pd.DataFrame:
    """Compute tokens/stems columns for the selected text and return the augmented dataframe."""
    d = df.copy()
    texts, _ = select_text_column(d)
    d["text_for_preprocess"] = texts
    d = d[d["text_for_preprocess"].str.strip().str.len() > 0].copy()
    print(f"Loaded rows (non-empty {CONFIG['TEXT_SOURCE']}): {len(d)}")

    out = d["text_for_preprocess"].apply(lambda x: preprocess_two_versions(x, stopwords_set, stemmer, min_len))
    d["tokens_clean_str"] = out.apply(lambda x: x["tokens_clean_str"])
    d["stems_str"] = out.apply(lambda x: x["stems_str"])
    d["n_tokens_clean"] = out.apply(lambda x: x["n_tokens_clean"])
    d["n_stems"] = out.apply(lambda x: x["n_stems"])
    return d


def write_output(df: pd.DataFrame, out_path: Path) -> None:
    """Write the preprocessed dataset to CSV with standard metadata and text fields."""
    keep_meta = [c for c in ["date", "url", "title", "subtitle", "method"] if c in df.columns]
    _, raw_col = select_text_column(df)

    keep_cols = keep_meta + [
        raw_col,
        "n_tokens_clean",
        "n_stems",
        "tokens_clean_str",
        "stems_str",
    ]
    df[keep_cols].to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved CSV: {out_path}")


def print_preview(df: pd.DataFrame) -> None:
    """Print a short preview ordered by date (first 3 rows)."""
    tmp = df.copy()
    tmp["date_dt"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp[tmp["date_dt"].notna()].sort_values("date_dt").head(3)

    if len(tmp):
        print("Sample (date, n_tokens_clean, first 12 stems):")
        for _, row in tmp.iterrows():
            stems_preview = " ".join(str(row["stems_str"]).split()[:12])
            print(row["date"], int(row["n_tokens_clean"]), stems_preview)
    else:
        print("Sample: no valid dates available to preview ordering.")


def main() -> None:
    """Execute preprocessing and write the final dataset."""
    project_root = get_project_root()
    in_path, out_path = resolve_paths(project_root)

    df = load_input(in_path)
    df = fill_date_string(df, load_date_map(project_root))

    stopwords_set, sw_src = get_stopwords()
    stemmer, stem_src = get_stemmer()
    print(f"Stopwords: {sw_src} | Stemmer: {stem_src}")

    df = add_preprocessed_columns(df, stopwords_set, stemmer, CONFIG["MIN_TOKEN_LEN"])
    write_output(df, out_path)
    print_preview(df)


if __name__ == "__main__":
    main()