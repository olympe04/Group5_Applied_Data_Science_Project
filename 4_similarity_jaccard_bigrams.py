# 04_similarity_jaccard_bigrams_csv.py
# Compute similarity between consecutive ECB statements using Jaccard similarity on bigrams
# from stems_str (space-joined stems) produced by 03_preprocess_csv_only.py
#
# Input : data_clean/ecb_statements_preprocessed.csv
# Output:
#   - data_clean/ecb_similarity_jaccard_bigrams.csv
#
# Notes:
# - Uses stems_str.split() -> no list parsing / no parquet dependency.
# - Dedupe by date: keeps the row with the largest n_stems (most content) for each day.

from pathlib import Path
import pandas as pd


def bigram_set(tokens):
    if not isinstance(tokens, list) or len(tokens) < 2:
        return set()
    return set(zip(tokens[:-1], tokens[1:]))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    u = a | b
    if not u:
        return float("nan")
    return len(a & b) / len(u)


def main():
    base_dir = Path(__file__).resolve().parent
    data_clean = base_dir / "data_clean"

    in_path = data_clean / "ecb_statements_preprocessed.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path} (run step 03 CSV-only first)")

    df = pd.read_csv(in_path)

    # Required columns
    for col in ["date", "url", "stems_str"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' in {in_path}. Expected output from step 03 CSV-only.")

    df = df.copy()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["date"] = df["date"].fillna("").astype(str).str.slice(0, 10)

    # Parse date for ordering
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"].notna()].copy()

    # Ensure stems_str present
    df["stems_str"] = df["stems_str"].fillna("").astype(str)
    df = df[df["stems_str"].str.len() > 0].copy()

    # Dedupe same date: keep the most content
    if "n_stems" in df.columns:
        df["len_for_dedupe"] = pd.to_numeric(df["n_stems"], errors="coerce").fillna(0).astype(int)
    else:
        df["len_for_dedupe"] = df["stems_str"].apply(lambda s: len(s.split()))

    df = (
        df.sort_values(["date_dt", "len_for_dedupe"], ascending=[True, False])
          .drop_duplicates(subset=["date_dt"], keep="first")
          .sort_values("date_dt")
          .reset_index(drop=True)
    )

    print(f"Rows after date filter + dedupe: {len(df)}")
    print(f"Date range: {df['date_dt'].min().date()} -> {df['date_dt'].max().date()}")

    if len(df) < 2:
        raise ValueError("Need at least 2 dated statements to compute consecutive similarity.")

    # Build bigram sets from stems_str
    df["stems_list"] = df["stems_str"].apply(lambda s: s.split())
    df["bigrams"] = df["stems_list"].apply(bigram_set)
    df["n_bigrams"] = df["bigrams"].apply(len)

    # Similarity vs previous
    sims = [float("nan")]
    prev_dates = [None]
    inter_sizes = [None]
    union_sizes = [None]

    for i in range(1, len(df)):
        a = df.at[i, "bigrams"]
        b = df.at[i - 1, "bigrams"]
        sims.append(jaccard(a, b))
        prev_dates.append(df.at[i - 1, "date"])
        inter_sizes.append(len(a & b))
        union_sizes.append(len(a | b))

    df["prev_date"] = prev_dates
    df["sim_jaccard_bigrams"] = sims
    df["intersection_size"] = inter_sizes
    df["union_size"] = union_sizes

    # Output table
    out_cols = ["date", "prev_date", "sim_jaccard_bigrams", "n_bigrams",
                "intersection_size", "union_size", "url"]
    if "title" in df.columns:
        out_cols.insert(-1, "title")

    out = df[out_cols].copy()

    out_path = data_clean / "ecb_similarity_jaccard_bigrams.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")

    print("\nPreview (first 8 rows):")
    print(out.head(8)[["date", "prev_date", "sim_jaccard_bigrams", "n_bigrams"]])

    print("\nPreview (last 5 rows):")
    print(out.tail(5)[["date", "prev_date", "sim_jaccard_bigrams", "n_bigrams"]])


if __name__ == "__main__":
    main()
