# E6_sentiment_pessimism_lm.py
# Extension of step 6: Compute ECB pessimism using Loughran–McDonald dictionary
# + split pessimism into bad news vs good news components (asymmetry / non-linearity).
#
# Pessimism = (Negative - Positive) / TotalWords.
# Bad news component  (pess_neg) = max(Pessimism, 0)
# Good news component (pess_pos) = min(Pessimism, 0)
#
# I/O:
#   Inputs: data_clean/ecb_statements_preprocessed.csv (requires tokens_clean_str)
#           data_raw/Loughran-McDonald_MasterDictionary_1993-2024.csv
#   Outputs: data_features/ecb_pessimism_lm.csv
#            data_features/ecb_statements_with_pessimism.csv
#            (optional) outputs/plots/pessimism_lm_<START>_<END>.png

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


CONFIG = {
    "INPUT_CSV": "data_clean/ecb_statements_preprocessed.csv",
    "LM_DICTIONARY_CSV": "data_raw/Loughran-McDonald_MasterDictionary_1993-2024.csv",
    "OUTPUT_PESSIMISM_CSV": "data_features/ecb_pessimism_lm.csv",                 # <- changed
    "OUTPUT_MERGED_CSV": "data_features/ecb_statements_with_pessimism.csv",       # <- changed
    "OUTPUT_DIR": "outputs/plots",
    "DEFAULT_START_DATE": "1999-01-01",
    "DEFAULT_END_DATE": "2013-12-31",
    "PLOT": True,
    "SHOW_PLOT": True,
    "PLOT_DPI": 200,
    "ADD_PCT_VERSION": True,
}


def get_project_root() -> Path:
    """
    Robustly find project root even if this script is moved into subfolders like:
      <root>/extension/separate_goodnews_badnews/E6_sentiment_pessimism_lm.py

    Strategy: walk up parents until we find expected root markers.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError(
        "Could not locate project root. Expected to find 'data_clean/' and 'outputs/' in a parent directory."
    )


def get_window_from_env() -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """Return (start_dt, end_dt, start_str, end_str) from env (or defaults)."""
    start_str = os.getenv("ECB_START_DATE", CONFIG["DEFAULT_START_DATE"])
    end_str = os.getenv("ECB_END_DATE", CONFIG["DEFAULT_END_DATE"])
    start_dt = pd.Timestamp(start_str)
    end_dt = pd.Timestamp(end_str)
    if end_dt < start_dt:
        raise ValueError(f"Invalid window: end < start ({start_str} .. {end_str})")
    return start_dt, end_dt, start_str, end_str


def resolve_paths(project_root: Path) -> dict[str, Path]:
    """Build all input/output paths used by the script."""
    return {
        "in_csv": project_root / CONFIG["INPUT_CSV"],
        "lm_csv": project_root / CONFIG["LM_DICTIONARY_CSV"],
        "out_pess": project_root / CONFIG["OUTPUT_PESSIMISM_CSV"],
        "out_merged": project_root / CONFIG["OUTPUT_MERGED_CSV"],
        "out_plots": project_root / CONFIG["OUTPUT_DIR"],
    }


def validate_inputs(paths: dict[str, Path]) -> None:
    """Check required input files exist before processing."""
    if not paths["in_csv"].exists():
        raise FileNotFoundError(
            f"Missing input: {paths['in_csv']}\n"
            "Run preprocess first to generate ecb_statements_preprocessed.csv"
        )
    if not paths["lm_csv"].exists():
        raise FileNotFoundError(f"Missing LM dictionary: {paths['lm_csv']}")


def load_preprocessed(in_path: Path) -> pd.DataFrame:
    """Load preprocessed statements and validate required columns."""
    df = pd.read_csv(in_path)
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column in ecb_statements_preprocessed.csv")
    if "tokens_clean_str" not in df.columns:
        raise ValueError(
            "Missing 'tokens_clean_str' in preprocessed data.\n"
            "LM pessimism must be computed on cleaned tokens (tokens_clean_str), not stems."
        )
    return df


def preprocess_statements(df: pd.DataFrame, text_col: str = "tokens_clean_str") -> pd.DataFrame:
    """Parse dates, drop invalid rows, and deduplicate by date (keep longest text)."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d["date"].notna()].copy()
    if len(d) == 0:
        raise ValueError("No valid dated rows in input after parsing 'date'.")

    d[text_col] = d[text_col].fillna("").astype(str)
    d["len_for_dedupe"] = d[text_col].apply(lambda s: len(s.split()))
    d = (
        d.sort_values(["date", "len_for_dedupe"], ascending=[True, False])
        .drop_duplicates(subset=["date"], keep="first")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return d


def load_lm_sets(lm_path: Path) -> tuple[set[str], set[str]]:
    """Load LM dictionary and return lowercase sets of negative and positive words."""
    lm = pd.read_csv(lm_path)

    if "Word" not in lm.columns:
        raise ValueError(f"LM dictionary missing 'Word' column: {lm_path}")
    if "Negative" not in lm.columns or "Positive" not in lm.columns:
        raise ValueError(f"LM dictionary missing 'Negative'/'Positive' columns: {lm_path}")

    lm["Word"] = lm["Word"].astype(str).str.strip().str.lower()
    neg_set = set(lm.loc[pd.to_numeric(lm["Negative"], errors="coerce").fillna(0).astype(int) > 0, "Word"])
    pos_set = set(lm.loc[pd.to_numeric(lm["Positive"], errors="coerce").fillna(0).astype(int) > 0, "Word"])
    return neg_set, pos_set


def count_sentiment(tokens_str: str, neg_set: set[str], pos_set: set[str]) -> tuple[int, int, int, float | None]:
    """Return (neg, pos, total, pessimism) for one token string."""
    if not isinstance(tokens_str, str) or tokens_str.strip() == "":
        return 0, 0, 0, None

    tokens = tokens_str.split()
    total = len(tokens)
    if total == 0:
        return 0, 0, 0, None

    c = Counter(tokens)
    neg = sum(freq for w, freq in c.items() if w in neg_set)
    pos = sum(freq for w, freq in c.items() if w in pos_set)
    return neg, pos, total, (neg - pos) / total


def compute_pessimism(df: pd.DataFrame, neg_set: set[str], pos_set: set[str], text_col: str) -> pd.DataFrame:
    """Add LM counts and pessimism columns (+ asymmetry split) to the dataframe."""
    d = df.copy()
    res = d[text_col].apply(lambda s: count_sentiment(s, neg_set, pos_set))

    d["lm_negative"] = res.apply(lambda x: x[0])
    d["lm_positive"] = res.apply(lambda x: x[1])
    d["lm_total"] = res.apply(lambda x: x[2])
    d["pessimism_lm"] = res.apply(lambda x: x[3])

    # ---- Extension: asymmetry split (bad vs good news) ----
    d["pess_neg_lm"] = d["pessimism_lm"].clip(lower=0)  # bad news component: >=0
    d["pess_pos_lm"] = d["pessimism_lm"].clip(upper=0)  # good news component: <=0

    if CONFIG["ADD_PCT_VERSION"]:
        d["pessimism_lm_pct"] = d["pessimism_lm"] * 100.0
        d["pess_neg_lm_pct"] = d["pess_neg_lm"] * 100.0
        d["pess_pos_lm_pct"] = d["pess_pos_lm"] * 100.0

    return d


def save_outputs(df_all: pd.DataFrame, out_pess: Path, out_merged: Path) -> None:
    """Write the full-sample pessimism file and the merged statements file."""
    keep_meta = [c for c in ["date", "url", "title", "subtitle", "method"] if c in df_all.columns]

    out_cols = keep_meta + [
        "lm_negative",
        "lm_positive",
        "lm_total",
        "pessimism_lm",
        "pess_neg_lm",
        "pess_pos_lm",
    ]
    if CONFIG["ADD_PCT_VERSION"]:
        out_cols += ["pessimism_lm_pct", "pess_neg_lm_pct", "pess_pos_lm_pct"]

    out_pess.parent.mkdir(parents=True, exist_ok=True)
    out_df = df_all[out_cols].copy()
    out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out_df.to_csv(out_pess, index=False, encoding="utf-8")
    print(f"Saved FULL sample: {out_pess}")

    out_merged.parent.mkdir(parents=True, exist_ok=True)
    merged = df_all.copy()
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged.to_csv(out_merged, index=False, encoding="utf-8")
    print(f"Saved merged FULL sample: {out_merged}")


def plot_window(df_all: pd.DataFrame, out_dir: Path) -> None:
    """Plot pessimism over the env window and save PNG to outputs/plots."""
    start_dt, end_dt, start_str, end_str = get_window_from_env()

    w = df_all[
        (df_all["date"] >= start_dt)
        & (df_all["date"] <= end_dt)
        & (df_all["pessimism_lm"].notna())
    ].copy()

    if len(w) == 0:
        raise ValueError(f"No valid pessimism values to plot in window {start_str}–{end_str}.")

    out_dir.mkdir(parents=True, exist_ok=True)
    w = w.sort_values("date")
    print(f"Plot observations: {len(w)}")

    plt.figure()
    plt.plot(w["date"], w["pessimism_lm"], linewidth=1)
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


def main() -> None:
    """Execute the full pessimism pipeline (with asymmetry split)."""
    project_root = get_project_root()
    paths = resolve_paths(project_root)
    validate_inputs(paths)

    print(f"Using input: {paths['in_csv']}")
    print(f"Using LM dict: {paths['lm_csv'].name}")

    df_raw = load_preprocessed(paths["in_csv"])
    df_all = preprocess_statements(df_raw, text_col="tokens_clean_str")
    print(f"Rows in FULL sample (post-dedupe): {len(df_all)}")

    neg_set, pos_set = load_lm_sets(paths["lm_csv"])
    df_all = compute_pessimism(df_all, neg_set, pos_set, text_col="tokens_clean_str")

    save_outputs(df_all, paths["out_pess"], paths["out_merged"])

    if CONFIG["PLOT"]:
        plot_window(df_all, paths["out_plots"])


if __name__ == "__main__":
    main()
