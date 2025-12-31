# 4_preprocess.py
# Preprocess ECB statements (CSV-only, robust):
# - Two parallel versions:
#   (1) tokens_clean (no stemming) -> for sentiment lexicons
#   (2) stems (Porter)             -> for similarity (Jaccard bigrams)
#
# Date handling:
# - If 'date' missing/unparseable in ecb_statements_raw.csv, merge from data_raw/ecb_speech_urls.txt (date,url)
#
# Input : data_raw/ecb_statements_raw.csv
# Output: data_clean/ecb_statements_preprocessed.csv  (CSV ONLY)
#
# Output columns include:
# - date (YYYY-MM-DD)
# - url, title, subtitle, method (if present)
# - statement_text
# - tokens_clean_str  (space-joined)
# - stems_str         (space-joined)
# - n_tokens_clean, n_stems

import re
from pathlib import Path
import pandas as pd


# Stopwords + Stemmer loaders

def _get_stopwords():
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english")), "nltk"
    except Exception:
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            return set(ENGLISH_STOP_WORDS), "sklearn"
        except Exception:
            return {
                "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "to", "of", "in", "on",
                "with", "as", "by", "at", "from", "into", "over", "under", "after", "before", "between",
                "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those",
                "we", "you", "they", "i", "he", "she", "them", "us", "our", "your", "their"
            }, "minimal"


def _get_stemmer():
    try:
        from nltk.stem import PorterStemmer
        return PorterStemmer(), "nltk_porter"
    except Exception:
        class NoOpStemmer:
            def stem(self, w: str) -> str:
                return w
        return NoOpStemmer(), "noop"


# Tokenization + cleaning

TOKEN_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*")
WHITESPACE_RE = re.compile(r"\s+")
NONBREAKING_RE = re.compile(r"\xa0")


def clean_text_basic(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    t = NONBREAKING_RE.sub(" ", t)
    t = t.replace("\u2019", "'")
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text: str):
    t = clean_text_basic(text).lower()
    return TOKEN_RE.findall(t)


def remove_stopwords(tokens, stopwords_set: set, min_len: int = 2):
    return [w for w in tokens if len(w) >= min_len and w not in stopwords_set]


def preprocess_two_versions(text: str, stopwords_set: set, stemmer) -> dict:
    tokens_raw = tokenize(text)
    tokens_clean = remove_stopwords(tokens_raw, stopwords_set=stopwords_set, min_len=2)
    stems = [stemmer.stem(w) for w in tokens_clean]
    return {
        "tokens_clean": tokens_clean,
        "stems": stems,
        "tokens_clean_str": " ".join(tokens_clean),
        "stems_str": " ".join(stems),
        "n_tokens_clean": len(tokens_clean),
        "n_stems": len(stems),
    }


# Date map (date,url) merge

def load_date_map(base_dir: Path) -> pd.DataFrame:
    data_raw = base_dir / "data_raw"
    candidates = [
        data_raw / "ecb_speech_urls.txt",
        data_raw / "ecb_speech_urls.csv",
    ]
    for p in candidates:
        if p.exists():
            dm = pd.read_csv(p)
            cols = {c.lower(): c for c in dm.columns}
            if "url" in cols and "date" in cols:
                out = dm[[cols["url"], cols["date"]]].copy()
                out.columns = ["url", "date_map"]
            else:
                if dm.shape[1] < 2:
                    continue
                # assume date,url in first two columns
                tmp = dm.iloc[:, :2].copy()
                tmp.columns = ["date_map", "url"]
                out = tmp[["url", "date_map"]]

            out["url"] = out["url"].astype(str).str.strip()
            out["date_map"] = out["date_map"].astype(str).str.slice(0, 10)
            return out

    return pd.DataFrame(columns=["url", "date_map"])


def fill_date_string(df: pd.DataFrame, date_map: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["url"] = df["url"].fillna("").astype(str).str.strip()

    if "date" not in df.columns:
        df["date"] = ""
    df["date"] = df["date"].fillna("").astype(str).str.slice(0, 10)

    # parse check
    parsed = pd.to_datetime(df["date"], errors="coerce")
    print(f"Initial date parse: non-NaT = {parsed.notna().sum()} / {len(df)}")

    if len(date_map) == 0:
        return df

    df = df.merge(date_map, on="url", how="left")

    need_fill = parsed.isna()
    df.loc[need_fill, "date"] = df.loc[need_fill, "date_map"].fillna("").astype(str).str.slice(0, 10)

    parsed2 = pd.to_datetime(df["date"], errors="coerce")
    print(f"After fill:        non-NaT = {parsed2.notna().sum()} / {len(df)}")

    # drop helper col
    if "date_map" in df.columns:
        df = df.drop(columns=["date_map"])

    return df


def main():
    base_dir = Path(__file__).resolve().parent
    data_raw = base_dir / "data_raw"
    data_clean = base_dir / "data_clean"
    data_clean.mkdir(parents=True, exist_ok=True)

    in_path = data_raw / "ecb_statements_raw_filtered.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Input introuvable: {in_path}")

    df = pd.read_csv(in_path)

    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.len().eq(0)].copy()

    if "url" not in df.columns:
        raise ValueError("Input missing 'url' column in ecb_statements_raw.csv")

    df["statement_text"] = df.get("statement_text", "").fillna("").astype(str)
    df = df[df["statement_text"].str.strip().str.len() > 0].copy()

    print(f"Loaded rows (non-empty statements): {len(df)}")

    date_map = load_date_map(base_dir)
    if len(date_map) > 0:
        print(f"Loaded date map rows: {len(date_map)}")
    else:
        print("No date map found (ecb_speech_urls.*). Dates will rely on scrape output only.")

    df = fill_date_string(df, date_map)

    stopwords_set, sw_src = _get_stopwords()
    stemmer, stem_src = _get_stemmer()
    print(f"Stopwords source: {sw_src} | Stemmer: {stem_src}")

    out = df["statement_text"].apply(lambda x: preprocess_two_versions(x, stopwords_set, stemmer))

    df["tokens_clean_str"] = out.apply(lambda d: d["tokens_clean_str"])
    df["stems_str"] = out.apply(lambda d: d["stems_str"])
    df["n_tokens_clean"] = out.apply(lambda d: d["n_tokens_clean"])
    df["n_stems"] = out.apply(lambda d: d["n_stems"])

    # Keep one date column (string YYYY-MM-DD)
    keep_meta = [c for c in ["date", "url", "title", "subtitle", "method"] if c in df.columns]
    keep_cols = keep_meta + [
        "statement_text",
        "n_tokens_clean", "n_stems",
        "tokens_clean_str", "stems_str",
    ]

    out_csv = data_clean / "ecb_statements_preprocessed.csv"
    df[keep_cols].to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved CSV: {out_csv}")

    # Preview
    tmp = df[keep_cols].copy()
    tmp["date_dt"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp[tmp["date_dt"].notna()].sort_values("date_dt").head(3)
    print("Sample (date, n_tokens_clean, first 12 stems):")
    for _, row in tmp.iterrows():
        stems_preview = " ".join(row["stems_str"].split()[:12])
        print(row["date"], int(row["n_tokens_clean"]), stems_preview)


if __name__ == "__main__":
    main()
