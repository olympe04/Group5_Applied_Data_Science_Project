# 03_preprocess.py
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def basic_preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"(http\S+|www\.\S+)", " ", text)   # URLs
    text = re.sub(r"[^a-z\s'-]", " ", text)          # keep letters, spaces, ' and -
    text = re.sub(r"\s+", " ", text).strip()

    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


def trim_words(text: str, max_words: int) -> str:
    """
    Optional: keep only the first max_words words to avoid extremely long Q&A
    dominating computation / memory. For TF-IDF cosine, it is usually fine,
    but trimming makes things more stable and reproducible.
    """
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


if __name__ == "__main__":
    in_path = "data_raw/ecb_statements_raw.csv"
    df = pd.read_csv(in_path)

    # --- Dates: parse + sort (important for time series)
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column in data_raw/ecb_statements_raw.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["date_iso"] = df["date"].dt.strftime("%Y-%m-%d")

    # --- Ensure text columns exist
    for col in ["statement_text", "qa_text"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # --- Word counts (raw)
    df["statement_words_raw"] = df["statement_text"].str.split().str.len()
    df["qa_words_raw"] = df["qa_text"].str.split().str.len()

    # --- Optional trimming (keeps things stable; adjust or set very high if you prefer no trim)
    # Statement is short; Q&A is long. Keep enough to preserve content while avoiding extreme tails.
    df["statement_text_trim"] = df["statement_text"].apply(lambda x: trim_words(x, 4000))
    df["qa_text_trim"] = df["qa_text"].apply(lambda x: trim_words(x, 8000))

    # --- Clean text (baseline vs extension)
    df["statement_clean"] = df["statement_text_trim"].apply(basic_preprocess)
    df["qa_clean"] = df["qa_text_trim"].apply(basic_preprocess)

    # Backward compatibility (if later scripts expect this)
    df["text_clean"] = df["statement_clean"]

    # Word counts (clean)
    df["statement_words_clean"] = df["statement_clean"].str.split().str.len()
    df["qa_words_clean"] = df["qa_clean"].str.split().str.len()

    # --- Save
    os.makedirs("data_clean", exist_ok=True)
    out_path = "data_clean/ecb_statements_clean.csv"
    df.to_csv(out_path, index=False)

    # --- Sanity checks
    print(f"Loaded:  {in_path}")
    print(f"Saved:   {out_path}")
    print(f"Rows:    {len(df)}")
    print("Date min:", df["date"].min().date() if len(df) else None)
    print("Date max:", df["date"].max().date() if len(df) else None)
    print("Empty statements (clean):", int((df["statement_clean"].str.len() == 0).sum()))
    print("Empty Q&A (clean):", int((df["qa_clean"].str.len() == 0).sum()))
    print("Mean words (statement raw):", round(df["statement_words_raw"].mean(), 1))
    print("Mean words (Q&A raw):", round(df["qa_words_raw"].mean(), 1))
