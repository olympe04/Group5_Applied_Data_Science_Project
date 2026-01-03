# extension/information_structure/main.py
# Run ONLY the information-structure extension pipeline
#
# Steps:
#   (1) E10_information_structure_features.py
#   (2) E11_regression_information_structure.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CONFIG = {
    "START_DATE": "1999-01-01",
    "END_DATE": "2025-12-31",
    "RUN_FEATURES": True,
    "RUN_REGRESSIONS": True,
    "DRY_RUN": False,
}

STEPS = [
    ("features", "E10_information_structure_features.py", CONFIG["RUN_FEATURES"]),
    ("regressions", "E11_regression_information_structure.py", CONFIG["RUN_REGRESSIONS"]),
]


def run_script(script_path: Path, project_root: Path, env: dict[str, str]) -> None:
    cmd = [sys.executable, str(script_path)]
    print(f"▶ {script_path.name}")

    if CONFIG["DRY_RUN"]:
        print("  ", " ".join(cmd))
        return

    subprocess.run(cmd, check=True, cwd=str(project_root), env={**os.environ, **env})


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent      # .../extension/information_structure
    project_root = scripts_dir.parent.parent          # .../project_root

    selected = [script for _, script, on in STEPS if on]
    if not selected:
        print("No steps selected (all RUN_* are False). Nothing to do.")
        return

    env = {
        "ECB_START_DATE": CONFIG["START_DATE"],
        "ECB_END_DATE": CONFIG["END_DATE"],
    }

    for script in selected:
        run_script((scripts_dir / script).resolve(), project_root, env)

    print("Done.")


if __name__ == "__main__":
    main()
