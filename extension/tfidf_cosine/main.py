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
    # Which steps to run:
    "RUN_SIMILARITY": True,      # E5_tfidf_cosine.py
    "RUN_MERGE_DATASET": False,  # ../E8_merge_car_pessimism_similarity.py (optional)
    "RUN_REGRESSIONS": True,     # E9_regression_tfidf.py
    # Execution mode:
    "DRY_RUN": False,
}

STEPS = [
    ("tfidf_similarity", "E5_tfidf_cosine.py", CONFIG["RUN_SIMILARITY"]),
    ("merge_analysis_dataset", "../E8_merge_car_pessimism_similarity.py", CONFIG["RUN_MERGE_DATASET"]),
    ("regressions_tfidf", "E9_regression_tfidf.py", CONFIG["RUN_REGRESSIONS"]),
]


def get_project_root() -> Path:
    """Find repo root from extension/tfidf_cosine/ by walking up until data_clean/ and outputs/ exist."""
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError("Could not locate project root (expected 'data_clean/' and 'outputs/' in a parent dir).")


def require_file(root: Path, relpath: str, hint: str) -> None:
    p = root / relpath
    if not p.exists():
        raise FileNotFoundError(f"Missing required input:\n  {p}\n{hint}")


def validate_prereqs(root: Path) -> None:
    if CONFIG["RUN_SIMILARITY"]:
        require_file(
            root,
            "data_clean/ecb_statements_preprocessed.csv",
            "Run replication preprocess (step 4) to generate ecb_statements_preprocessed.csv.",
        )

    if CONFIG["RUN_MERGE_DATASET"] or CONFIG["RUN_REGRESSIONS"]:
        require_file(
            root,
            "data_clean/ecb_pessimism_with_car.csv",
            "Run replication steps 6 + 7 to generate ecb_pessimism_with_car.csv.",
        )

    if CONFIG["RUN_REGRESSIONS"]:
        require_file(
            root,
            "data_clean/controls_month_end.csv",
            "Run replication step 7b to generate controls_month_end.csv.",
        )


def run_script(script_path: Path, project_root: Path, env: dict[str, str]) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [sys.executable, str(script_path)]
    print(f"▶ {script_path.name}")

    if CONFIG["DRY_RUN"]:
        print("  ", " ".join(cmd))
        return

    subprocess.run(cmd, check=True, cwd=str(project_root), env={**os.environ, **env})


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent  # .../extension/tfidf_cosine
    project_root = get_project_root()

    selected = [script for _, script, on in STEPS if on]
    if not selected:
        print("No steps selected (all RUN_* are False). Nothing to do.")
        return

    env = {
        "ECB_START_DATE": CONFIG["START_DATE"],
        "ECB_END_DATE": CONFIG["END_DATE"],
    }

    validate_prereqs(project_root)

    for script in selected:
        run_script((scripts_dir / script).resolve(), project_root, env)

    print("Done.")


if __name__ == "__main__":
    main()
