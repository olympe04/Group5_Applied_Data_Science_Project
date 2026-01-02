# 6_sentiment_pessimism_lm.py
# Compute ECB pessimism using the Loughran–McDonald dictionary:
#   Pessimism = (Negative - Positive) / TotalWords.
# I/O:
#   Inputs: data_clean/ecb_statements_preprocessed.csv (requires tokens_clean_str) and data_raw/Loughran-McDonald_MasterDictionary_1993-2024.csv.
#   Outputs: data_clean/ecb_pessimism_lm.csv, data_clean/ecb_statements_with_pessimism.csv, and (optional) outputs/pessimism_lm_<START>_<END>.png.

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONFIG = {
    "INPUT_CSV": "data_clean/ecb_statements_preprocessed.csv",
    "LM_DICTIONARY_CSV": "data_raw/Loughran-McDonald_MasterDictionary_1993-2024.csv",
    "OUTPUT_PESSIMISM_CSV": "data_clean/ecb_pessimism_lm.csv",
    "OUTPUT_MERGED_CSV": "data_clean/ecb_statements_with_pessimism.csv",
    "OUTPUT_DIR": "outputs",
    "DEFAULT_START_DATE": "1999-01-01",
    "DEFAULT_END_DATE": "2013-12-31",
    "PLOT": True,
    "SHOW_PLOT": True,
    "PLOT_DPI": 200,
    "ADD_PCT_VERSION": True,
}


def get_window_from_env() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """Read (start, end) plot window from env (or defaults) and return parsed timestamps plus the original strings."""
    start_str = os.getenv("ECB_START_DATE", CONFIG["DEFAULT_START_DATE"])
    end_str = os.getenv("ECB_END_DATE", CONFIG["DEFAULT_END_DATE"])
    start_dt = pd.Timestamp(start_str)
    end_dt = pd.Timestamp(end_str)
    if end_dt < start_dt:
        raise ValueError(f"Invalid window: end < start ({start_str} .. {end_str})")
    return start_dt, end_dt, start_str, end_str


def load_lm_sets(lm_path: Path) -> tuple[set, set]:
    """Load the LM dictionary CSV and return lowercase sets of negative and positive words."""
    lm = pd.read_csv(lm_path)

    if "Word" not in lm.columns:
        raise ValueError(f"LM dictionary missing 'Word' column: {lm_path}")
    if "Negative" not in lm.columns or "Positive" not in lm.columns:
        raise ValueError(f"LM dictionary missing 'Negative'/'Positive' columns: {lm_path}")

    lm["Word"] = lm["Word"].astype(str).str.strip().str.lower()

    neg_set = set(lm.loc[pd.to_numeric(lm["Negative"], errors="coerce").fillna(0).astype(int) > 0, "Word"])
    pos_set = set(lm.loc[pd.to_numeric(lm["Positive"], errors="coerce").fillna(0).astype(int) > 0, "Word"])
    return neg_set, pos_set


def count_sentiment(tokens_str: str, neg_set: set, pos_set: set):
    """Count LM negative/positive tokens and return (neg, pos, total, pessimism) for a token string."""
    if not isinstance(tokens_str, str) or tokens_str.strip() == "":
        return 0, 0, 0, None

    tokens = tokens_str.split()
    total = len(tokens)
    if total == 0:
        return 0, 0, 0, None

    c = Counter(tokens)
    neg = sum(freq for w, freq in c.items() if w in neg_set)
    pos = sum(freq for w, freq in c.items() if w in pos_set)
    pessimism = (neg - pos) / total
    return neg, pos, total, pessimism


def main() -> None:
    """Compute LM-based pessimism for the full dataset, save outputs, and optionally plot within a date window."""
    scripts_dir = Path(__file__).resolve().parent   # .../replication
    project_root = scripts_dir.parent               # .../ (repo root)

    in_path = project_root / CONFIG["INPUT_CSV"]
    lm_path = project_root / CONFIG["LM_DICTIONARY_CSV"]

    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing input: {in_path}\n"
            "Run preprocess first to generate ecb_statements_preprocessed.csv"
        )

    if not lm_path.exists():
        raise FileNotFoundError(f"Missing LM dictionary: {lm_path}")

    print(f"Using input: {in_path}")
    print(f"Using LM dict: {lm_path.name}")

    df_all = pd.read_csv(in_path)

    if "date" not in df_all.columns:
        raise ValueError("Missing 'date' column in ecb_statements_preprocessed.csv")

    if "tokens_clean_str" not in df_all.columns:
        raise ValueError(
            "Missing 'tokens_clean_str' in preprocessed data.\n"
            "LM pessimism must be computed on cleaned tokens (tokens_clean_str), not stems."
        )

    df_all = df_all.copy()
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    df_all = df_all[df_all["date"].notna()].copy()

    text_col = "tokens_clean_str"
    df_all[text_col] = df_all[text_col].fillna("").astype(str)

    if len(df_all) == 0:
        raise ValueError("No valid dated rows in input after parsing 'date'.")

    df_all["len_for_dedupe"] = df_all[text_col].apply(lambda s: len(s.split()))
    df_all = (
        df_all.sort_values(["date", "len_for_dedupe"], ascending=[True, False])
              .drop_duplicates(subset=["date"], keep="first")
              .sort_values("date")
              .reset_index(drop=True)
    )
    print(f"Rows in FULL sample (post-dedupe): {len(df_all)}")

    neg_set, pos_set = load_lm_sets(lm_path)

    results = df_all[text_col].apply(lambda s: count_sentiment(s, neg_set, pos_set))
    df_all["lm_negative"] = results.apply(lambda x: x[0])
    df_all["lm_positive"] = results.apply(lambda x: x[1])
    df_all["lm_total"] = results.apply(lambda x: x[2])
    df_all["pessimism_lm"] = results.apply(lambda x: x[3])

    if CONFIG["ADD_PCT_VERSION"]:
        df_all["pessimism_lm_pct"] = df_all["pessimism_lm"] * 100.0

    keep_meta = [c for c in ["date", "url", "title", "subtitle", "method"] if c in df_all.columns]
    out_cols = keep_meta + ["lm_negative", "lm_positive", "lm_total", "pessimism_lm"]
    if CONFIG["ADD_PCT_VERSION"]:
        out_cols.append("pessimism_lm_pct")

    out_path = project_root / CONFIG["OUTPUT_PESSIMISM_CSV"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = df_all[out_cols].copy()
    out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved FULL sample: {out_path}")

    merged_path = project_root / CONFIG["OUTPUT_MERGED_CSV"]
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    df2 = df_all.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df2.to_csv(merged_path, index=False, encoding="utf-8")
    print(f"Saved merged FULL sample: {merged_path}")

    if CONFIG["PLOT"]:
        out_dir = project_root / CONFIG["OUTPUT_DIR"]
        out_dir.mkdir(parents=True, exist_ok=True)

        start_dt, end_dt, start_str, end_str = get_window_from_env()

        w_plot = df_all[
            (df_all["date"] >= start_dt) &
            (df_all["date"] <= end_dt) &
            (df_all["pessimism_lm"].notna())
        ].copy()

        if len(w_plot) == 0:
            raise ValueError(f"No valid pessimism values to plot in window {start_str}–{end_str}.")

        w_plot = w_plot.sort_values("date")
        print(f"Plot observations: {len(w_plot)}")

        plt.figure()
        plt.plot(w_plot["date"], w_plot["pessimism_lm"], linewidth=1)
        plt.title(f"ECB pessimism (Loughran–McDonald), {start_str}–{end_str}")
        plt.xlabel("Date")
        plt.ylabel("Pessimism")
        plt.tight_layout()

        fig_path = out_dir / f"pessimism_lm_{start_str.replace('-', '')}_{end_str.replace('-', '')}.png"
        plt.savefig(fig_path, dpi=int(CONFIG["PLOT_DPI"]))
        if CONFIG["SHOW_PLOT"]:
            plt.show()
        else:
            plt.close()

        print(f"Saved plot (window-only): {fig_path}")


if __name__ == "__main__":
    main()
