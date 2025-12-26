# 05_plot_similarity.py
import os
import pandas as pd
import matplotlib.pyplot as plt


def add_ma(series: pd.Series, window: int = 6, min_periods: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=min_periods).mean()


if __name__ == "__main__":
    os.makedirs("outputs/figures", exist_ok=True)

    in_path = "outputs/tables/similarity_series.csv"
    df = pd.read_csv(in_path)

    # Ensure date is datetime and sorted
    if "date" not in df.columns:
        raise ValueError("Column 'date' not found in similarity_series.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Check required columns
    for col in ["similarity_statement", "similarity_qa"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' in similarity_series.csv")

    # Moving averages
    df["statement_ma6"] = add_ma(df["similarity_statement"], window=6, min_periods=3)
    df["qa_ma6"] = add_ma(df["similarity_qa"], window=6, min_periods=3)

    plt.figure(figsize=(10, 6))

    # Raw series
    plt.plot(df["date"], df["similarity_statement"], label="Statement (raw)")
    plt.plot(df["date"], df["similarity_qa"], label="Q&A (raw)")

    # Smoothed series
    plt.plot(df["date"], df["statement_ma6"], label="Statement MA(6)")
    plt.plot(df["date"], df["qa_ma6"], label="Q&A MA(6)")

    plt.title("ECB communication similarity (TF-IDF cosine): Statement vs Q&A")
    plt.xlabel("Date")
    plt.ylabel("Cosine similarity")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    out_fig = "outputs/figures/similarity_statement_vs_qa.png"
    plt.savefig(out_fig, dpi=200)
    print(f"Saved {out_fig}")
