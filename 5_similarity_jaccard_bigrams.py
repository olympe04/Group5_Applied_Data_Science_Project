# 5_similarity_jaccard_bigrams.py
# Compute similarity between consecutive ECB statements using Jaccard similarity on bigrams
# from stems_str (space-joined stems) produced by 4_preprocess.py
# Input:  data_clean/ecb_statements_preprocessed_dropped.csv
# Output: plot for 1999-01-01 -> 2013-12-31 (time series only)

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


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

    in_path = data_clean / "ecb_statements_preprocessed_dropped.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path} (run preprocess + drop step first).")

    print(f"Using input: {in_path.name}")
    df = pd.read_csv(in_path)

    # Required columns
    for col in ["date", "stems_str"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' in {in_path}. Expected output from step 03 CSV-only.")

    df = df.copy()
    df["date"] = df["date"].fillna("").astype(str).str.slice(0, 10)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"].notna()].copy()

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

    if len(df) < 2:
        raise ValueError("Need at least 2 dated statements to compute consecutive similarity.")

    # Build bigram sets and compute similarity vs previous
    df["stems_list"] = df["stems_str"].str.split()
    df["bigrams"] = df["stems_list"].apply(lambda toks: set(zip(toks[:-1], toks[1:])) if len(toks) >= 2 else set())

    sims = [float("nan")]
    for i in range(1, len(df)):
        sims.append(jaccard(df.at[i, "bigrams"], df.at[i - 1, "bigrams"]))

    df["sim_jaccard_bigrams"] = sims

    # Window 1999-01-01 -> 2013-12-31
    start = pd.Timestamp("1999-01-01")
    end = pd.Timestamp("2013-12-31")

    dfw = df[(df["date_dt"] >= start) & (df["date_dt"] <= end)].copy()
    dfw = dfw[dfw["sim_jaccard_bigrams"].notna()].sort_values("date_dt")

    if len(dfw) == 0:
        raise ValueError("No valid similarity values in the selected window 1999–2013.")

    # Plot time series
    plt.figure()
    plt.plot(dfw["date_dt"], dfw["sim_jaccard_bigrams"], linewidth=1)
    plt.title("ECB statement similarity (Jaccard bigrams), 1999–2013")
    plt.xlabel("Date")
    plt.ylabel("Similarity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
