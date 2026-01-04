# extension/separate_goodnews_badnews/main.py
# Run the "separate good news / bad news" extension pipeline in dependency order.
#
# Steps:
#   (1) E6_sentiment_pessimism_lm.py        -> creates data_clean/ecb_sentiment_lm.csv (+ plot)
#   (2) E7_event_study_car.py               -> creates data_clean/ecb_pessimism_with_car.csv
#   (3) E8_regressions-table4_asymmetry.py  -> creates outputs/tables/Etable4_absCAR_regressions_asymmetry.csv
#
# Prereqs expected (from replication / other extensions):
#   - data_clean/ecb_statements_preprocessed.csv
#   - data_raw/Loughran-McDonald_MasterDictionary_1993-2024.csv
#   - data_raw/^SX5E data.xlsx
#   - data_clean/controls_month_end.csv
#   - data_clean/ecb_similarity_jaccard_bigrams.csv   (required by E8; produced by your similarity extension)
#
# Env injected:
#   ECB_START_DATE / ECB_END_DATE  (used by E6 and E8)
#   ECB_ALIGN_MARKET               (used by E7_event_study_car.py: "next" or "previous")

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CONFIG = {
    "START_DATE": "1999-01-01",
    "END_DATE": "2013-12-31",
    "ALIGN_MARKET": "next",  # "next" or "previous"

    "RUN_LM_SENTIMENT": True,
    "RUN_EVENT_STUDY": True,
    "RUN_ASYM_REGS": True,
}

STEPS = [
    ("RUN_LM_SENTIMENT", "E6_sentiment_pessimism_lm.py"),
    ("RUN_EVENT_STUDY", "E7_event_study_car.py"),
    ("RUN_ASYM_REGS", "E8_regressions-table4_GBnews.py"),
]


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "data_clean").exists() and (p / "data_raw").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError("Could not locate project root (expected data_clean/, data_raw/, outputs/).")


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    root = get_project_root()

    selected = [name for flag, name in STEPS if CONFIG[flag]]
    if not selected:
        print("No steps selected (all RUN_* are False). Nothing to do.")
        return

    # prereqs expected (no fallbacks)
    required_inputs = [
        "data_clean/ecb_statements_preprocessed.csv",
        "data_raw/Loughran-McDonald_MasterDictionary_1993-2024.csv",
        "data_raw/^SX5E data.xlsx",
        "data_clean/controls_month_end.csv",
        "data_clean/ecb_similarity_jaccard_bigrams.csv",
    ]
    for rel in required_inputs:
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    env = {
        **os.environ,
        "ECB_START_DATE": CONFIG["START_DATE"],
        "ECB_END_DATE": CONFIG["END_DATE"],
        "ECB_ALIGN_MARKET": CONFIG["ALIGN_MARKET"],
    }

    for name in selected:
        script = (scripts_dir / name).resolve()
        if not script.exists():
            raise FileNotFoundError(f"Missing script: {script}")
        print(f"▶ {script.name}")
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(root), env=env)

    print("Done.")


if __name__ == "__main__":
    main()
