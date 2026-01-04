# extension/tfidf_cosine/main.py
# Run the TF-IDF cosine (consecutive cosine similarity) extension pipeline in dependency order.
# I/O:
#   Inputs : data_clean/ecb_statements_preprocessed.csv
#            data_clean/ecb_pessimism_with_car.csv
#            data_clean/controls_month_end.csv
#   Outputs: data_features/ecb_similarity_tfidf.csv (+ plot)
#            outputs/tables/table3_similarity_regressions_tfidf.csv
#            outputs/tables/table4_absCAR_regressions_tfidf.csv
#            outputs/tables/table4_absCAR_regressions_tfidf_z.csv
#            (optional) data_clean/ecb_analysis_dataset.csv (if enabled)
#
# Notes:
# - This runner assumes the replication pipeline already produced:
#     - data_clean/ecb_statements_preprocessed.csv
#     - data_clean/ecb_pessimism_with_car.csv
#     - data_clean/controls_month_end.csv
# - It injects ECB_START_DATE / ECB_END_DATE into subprocess env so scripts use the same window.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CONFIG = {
    "START_DATE": "1999-01-01",
    "END_DATE": "2013-12-31",

    "RUN_SIMILARITY": True,      # E5_tfidf_cosine.py
    "RUN_MERGE_DATASET": False,  # ../E8_merge_car_pessimism_similarity.py (optional)
    "RUN_REGRESSIONS": True,     # E9_regression_tfidf.py
}

STEPS = [
    ("RUN_SIMILARITY", "E5_tfidf_cosine.py"),
    ("RUN_MERGE_DATASET", "../E8_merge_car_pessimism_similarity.py"),
    ("RUN_REGRESSIONS", "E9_regression_tfidf.py"),
]


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError("Could not locate project root (expected data_clean/ and outputs/).")


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
        "data_clean/ecb_pessimism_with_car.csv",
        "data_clean/controls_month_end.csv",
    ]
    for rel in required_inputs:
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    env = {
        **os.environ,
        "ECB_START_DATE": CONFIG["START_DATE"],
        "ECB_END_DATE": CONFIG["END_DATE"],
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
