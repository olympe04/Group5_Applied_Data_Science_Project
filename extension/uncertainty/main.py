# extension/uncertainty/main.py
# Run ONLY the uncertainty extension pipeline
#
# Steps:
#   (1) E6_uncertainty_lm.py
#   (2) E7_event_study_car_uncertainty.py
#   (3) E9_regression_uncertainty.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CONFIG = {
    "START_DATE": "2014-01-01",
    "END_DATE": "2025-12-31",

    # toggle steps
    "RUN_UNCERTAINTY_SCORE": True,
    "RUN_EVENT_STUDY": True,
    "RUN_REGRESSIONS": True,

    # execution
    "DRY_RUN": False,
}

STEPS = [
    ("uncertainty_score", "E6_uncertainty.py", CONFIG["RUN_UNCERTAINTY_SCORE"]),
    ("event_study", "E7_event_study_car_uncertainty.py", CONFIG["RUN_EVENT_STUDY"]),
    ("regressions", "E9_regression_uncertainty.py", CONFIG["RUN_REGRESSIONS"]),
]


def run_script(script_path: Path, project_root: Path, env: dict[str, str]) -> None:
    cmd = [sys.executable, str(script_path)]
    print(f"▶ {script_path.name}")

    if CONFIG["DRY_RUN"]:
        print("  ", " ".join(cmd))
        return

    subprocess.run(cmd, check=True, cwd=str(project_root), env={**os.environ, **env})


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent    # .../extension/uncertainty
    project_root = scripts_dir.parent.parent        # .../project_root

    selected = [script for _, script, on in STEPS if on]
    if not selected:
        print("No steps selected (all RUN_* are False). Nothing to do.")
        return

    env = {
        "ECB_START_DATE": CONFIG["START_DATE"],
        "ECB_END_DATE": CONFIG["END_DATE"],
        "INDICATOR": "uncertainty",  # harmless even if E9 ignores it
    }

    for script in selected:
        run_script((scripts_dir / script).resolve(), project_root, env)

    print("Done.")


if __name__ == "__main__":
    main()