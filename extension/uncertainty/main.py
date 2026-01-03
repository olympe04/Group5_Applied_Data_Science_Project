# extension/uncertainty/main.py
# Run ONLY the uncertainty extension pipeline (no pessimism support, no INDICATOR switching).
#
# Steps:
#   (1) E6_uncertainty_lm.py
#   (2) E7_event_study_car_uncertainty.py
#   (3) E9_regression_uncertainty.py   (assumed to run uncertainty tables)

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


CONFIG = {
    "START_DATE": "1999-01-01",
    "END_DATE": "2025-12-31",

    # toggle steps
    "RUN_UNCERTAINTY_SCORE": True,
    "RUN_EVENT_STUDY": True,
    "RUN_REGRESSIONS": True,

    # execution
    "DRY_RUN": False,
}


@dataclass(frozen=True)
class Step:
    key: str
    script: str
    deps: Tuple[str, ...] = ()


PIPELINE: List[Step] = [
    Step("uncertainty_score", "E6_uncertainty.py"),
    Step("event_study", "E7_event_study_car_uncertainty.py", deps=("uncertainty_score",)),
    Step("regressions", "E9_regression_uncertainty.py", deps=("event_study",)),
]


def get_project_root() -> Path:
    """Find repo root from extension/uncertainty/ by walking up until data_clean/ and outputs/ exist."""
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError("Could not locate project root (expected 'data_clean/' and 'outputs/' in a parent dir).")


def expand_with_deps(wanted: Set[str], smap: Dict[str, Step]) -> List[Step]:
    """Resolve dependencies and preserve PIPELINE order."""
    seen: Set[str] = set()

    def add(k: str) -> None:
        if k not in smap:
            raise KeyError(f"Unknown step: {k}")
        if k in seen:
            return
        for d in smap[k].deps:
            add(d)
        seen.add(k)

    for k in list(wanted):
        add(k)

    return [s for s in PIPELINE if s.key in seen]


def require_file(project_root: Path, relpath: str, hint: str) -> None:
    p = project_root / relpath
    if not p.exists():
        raise FileNotFoundError(f"Missing required input:\n  {p}\n{hint}")


def validate_prereqs(project_root: Path, steps: List[Step]) -> None:
    keys = {s.key for s in steps}

    if "uncertainty_score" in keys:
        require_file(
            project_root,
            "data_clean/ecb_statements_preprocessed.csv",
            "Run replication preprocess (step 4) to generate ecb_statements_preprocessed.csv.",
        )
        require_file(
            project_root,
            "data_raw/Loughran-McDonald_MasterDictionary_1993-2024.csv",
            "Provide the LM dictionary under data_raw/ (or update E6_uncertainty_lm.py CONFIG).",
        )

    if "event_study" in keys:
        require_file(
            project_root,
            "data_raw/^SX5E data.xlsx",
            "Provide market data under data_raw/^SX5E data.xlsx (or update E7_event_study_car_uncertainty.py).",
        )
        # Produced by E6 in your migrated setup:
        require_file(
            project_root,
            "data_features/ecb_uncertainty_lm.csv",
            "Run E6_uncertainty_lm.py first to generate data_features/ecb_uncertainty_lm.csv.",
        )

    if "regressions" in keys:
        # IMPORTANT: these stay in data_clean (raw jaccard + month-end controls)
        require_file(
            project_root,
            "data_clean/ecb_similarity_jaccard_bigrams.csv",
            "Run similarity step (replication step 5) to generate ecb_similarity_jaccard_bigrams.csv.",
        )
        require_file(
            project_root,
            "data_clean/controls_month_end.csv",
            "Run controls step (replication step 7b) to generate controls_month_end.csv.",
        )
        # Produced by the event_study step dependency (in data_features after your migration):
        require_file(
            project_root,
            "data_features/ecb_uncertainty_with_car.csv",
            "Run E7_event_study_car_uncertainty.py first to generate data_features/ecb_uncertainty_with_car.csv.",
        )


def run_step(scripts_dir: Path, project_root: Path, step: Step, env: Dict[str, str], dry_run: bool) -> None:
    script_path = scripts_dir / step.script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [sys.executable, str(script_path)]
    print(f"▶ {step.key:18s}  {step.script}")

    if dry_run:
        print("  ", " ".join(cmd))
        return

    subprocess.run(cmd, check=True, cwd=str(project_root), env={**os.environ, **env})


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent  # .../extension/uncertainty
    project_root = get_project_root()

    smap = {s.key: s for s in PIPELINE}

    wanted: Set[str] = set()
    if CONFIG["RUN_UNCERTAINTY_SCORE"]:
        wanted.add("uncertainty_score")
    if CONFIG["RUN_EVENT_STUDY"]:
        wanted.add("event_study")
    if CONFIG["RUN_REGRESSIONS"]:
        wanted.add("regressions")

    if not wanted:
        print("No steps selected (all RUN_* are False). Nothing to do.")
        return

    steps = expand_with_deps(wanted, smap)

    env = {
        "ECB_START_DATE": CONFIG["START_DATE"],
        "ECB_END_DATE": CONFIG["END_DATE"],
        "INDICATOR": "uncertainty",  # harmless even if E9 ignores it
    }

    validate_prereqs(project_root, steps)

    for step in steps:
        run_step(scripts_dir, project_root, step, env, CONFIG["DRY_RUN"])

    print("Done.")


if __name__ == "__main__":
    main()
