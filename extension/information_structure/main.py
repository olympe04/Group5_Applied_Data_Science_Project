# extension/information_structure/main.py
# Runs:
#   (1) E10_information_structure_features.py
#   (2) E11_regression_information_structure.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CONFIG = {
    "START_DATE": "1999-01-01",
    "END_DATE": "2013-12-31",
    "RUN_FEATURES": True,
    "RUN_REGRESSIONS": True,
    "DRY_RUN": False,
}

STEPS = [
    ("E10_information_structure_features.py", "RUN_FEATURES"),
    ("E11_regression_information_structure.py", "RUN_REGRESSIONS"),
]


def run_script(script: Path, root: Path, env: dict[str, str]) -> None:
    cmd = [sys.executable, str(script)]
    print(f"▶ {script.name}")
    if CONFIG["DRY_RUN"]:
        print("  ", " ".join(cmd))
        return
    subprocess.run(cmd, check=True, cwd=str(root), env={**os.environ, **env})


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    root = scripts_dir.parent.parent

    selected = [name for name, flag in STEPS if CONFIG[flag]]
    if not selected:
        print("No steps selected. Nothing to do.")
        return

    env = {
        "ECB_START_DATE": CONFIG["START_DATE"],
        "ECB_END_DATE": CONFIG["END_DATE"],
    }

    for name in selected:
        script = (scripts_dir / name).resolve()
        if not script.exists():
            raise FileNotFoundError(f"Missing script: {script}")
        run_script(script, root, env)

    print("Done.")


if __name__ == "__main__":
    main()
