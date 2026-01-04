# extension/learning_speed/main.py
# Run the Jaccard learning-speed extension pipeline in dependency order.
#
# Steps:
#   (1) E5_jaccard_learning_speed.py      -> data_features/ecb_similarity_jaccard_learning.csv (+ plot)
#   (2) E5c_plot_learning_speed.py        -> outputs/plots/ts_*_jaccard.png
#   (3) E9b_regression_learning_speed.py  -> outputs/tables/table4_absCAR_regressions_centroidk_jaccard*.csv
#
# Prereqs expected:
#   - data_clean/ecb_statements_preprocessed.csv
#   - data_clean/ecb_similarity_jaccard_bigrams.csv
#   - data_clean/ecb_pessimism_with_car.csv
#   - data_clean/controls_month_end.csv
#
# Env injected:
#   ECB_START_DATE / ECB_END_DATE (used by E5, E5c, E9b if they read env)

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CONFIG = {
    "START_DATE": "1999-01-01",
    "END_DATE": "2013-12-31",

    "RUN_LEARNING": True,   # E5_jaccard_learning_speed.py
    "RUN_PLOTS": True,      # E5c_plot_learning_speed.py
    "RUN_REGS": True,       # E9b_regression_learning_speed.py
}

STEPS = [
    ("RUN_LEARNING", "E5_jaccard_learning_speed.py"),
    ("RUN_PLOTS", "E5c_plot_learning_speed.py"),
    ("RUN_REGS", "E9_regression_learning_speed.py"),
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
        "data_clean/ecb_similarity_jaccard_bigrams.csv",
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
