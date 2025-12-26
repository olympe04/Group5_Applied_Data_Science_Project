# 04_similarity.py
# Compute textual similarity over time
# Baseline: prepared statements
# Extension: Q&A section

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_series(texts: pd.Series) -> list:
    """
    Compute cosine similarity between consecutive documents.
    Returns a list of length N with NaN for the first observation.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=1
    )

    X = vectorizer.fit_transform(texts.fillna(""))

    sims = [np.nan]
    for i in range(1, X.shape[0]):
        sims.append(cosine_similarity(X[i], X[i - 1])[0, 0])

    return sims


if __name__ == "__main__":
    # --- Load clean data
    df = pd.read_csv("data_clean/ecb_statements_clean.csv")

    # --- Ensure date ordering (CRUCIAL)
    if "date" not in df.columns:
        raise ValueError("Column 'date' not found in clean data")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # --- Check required text columns
    for col in ["statement_clean", "qa_clean"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in clean data")

    # --- Compute similarity series
    print("Computing similarity for prepared statements (baseline)...")
    df["similarity_statement"] = compute_similarity_series(df["statement_clean"])

    print("Computing similarity for Q&A (extension)...")
    df["similarity_qa"] = compute_similarity_series(df["qa_clean"])

    # --- Save
    os.makedirs("outputs/tables", exist_ok=True)
    out_path = "outputs/tables/similarity_series.csv"
    df.to_csv(out_path, index=False)

    # --- Sanity checks
    print(f"Saved {out_path}")
    print("Observations:", len(df))
    print("Mean similarity (statement):",
          round(np.nanmean(df["similarity_statement"]), 3))
    print("Mean similarity (Q&A):",
          round(np.nanmean(df["similarity_qa"]), 3))
