# 04b_plot_similarity_1999_2013.py
# Visualisation de la similarité Jaccard (bigrams) sur 1999-01-01 -> 2013-12-31

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent
in_path = base_dir / "data_clean" / "ecb_similarity_jaccard_bigrams.csv"

df = pd.read_csv(in_path)

# Parse dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["prev_date"] = pd.to_datetime(df["prev_date"], errors="coerce")

# Filter window: 1999-01-01 to 2013-12-31
start = pd.Timestamp("1999-01-01")
end = pd.Timestamp("2013-12-31")
df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

# Keep valid sims (ignore first NaN)
df["sim_jaccard_bigrams"] = pd.to_numeric(df["sim_jaccard_bigrams"], errors="coerce")
df_valid = df[df["date"].notna() & df["sim_jaccard_bigrams"].notna()].sort_values("date").copy()

print("Window:", start.date(), "->", end.date())
print("N rows in window:", len(df))
print("N rows with similarity:", len(df_valid))

if len(df_valid) == 0:
    raise ValueError("No valid similarity values in the selected window. Check your dates/similarity file.")

print("Similarity min/mean/max:",
      df_valid["sim_jaccard_bigrams"].min(),
      df_valid["sim_jaccard_bigrams"].mean(),
      df_valid["sim_jaccard_bigrams"].max())

# 1) Time series
plt.figure()
plt.plot(df_valid["date"], df_valid["sim_jaccard_bigrams"], linewidth=1)
plt.title("ECB statement similarity (Jaccard bigrams), 1999–2013")
plt.xlabel("Date")
plt.ylabel("Similarity")
plt.tight_layout()
plt.show()

# 2) Histogram
plt.figure()
plt.hist(df_valid["sim_jaccard_bigrams"], bins=30)
plt.title("Distribution of similarity (Jaccard bigrams), 1999–2013")
plt.xlabel("Similarity")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 3) Biggest drops within window
df_valid["delta"] = df_valid["sim_jaccard_bigrams"].diff()
worst = df_valid.nsmallest(10, "delta")[["date", "prev_date", "sim_jaccard_bigrams", "delta"]]
print("\nTop 10 biggest drops in similarity (1999–2013):")
print(worst.to_string(index=False))
