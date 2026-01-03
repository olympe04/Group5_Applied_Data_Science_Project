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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


CONFIG = {
    "START_DATE": "1999-01-01",
    "END_DATE": "2025-12-31",
    # Which steps to run:
    "RUN_SIMILARITY": True,      # E5_tfidf_cosine.py
    "RUN_MERGE_DATASET": False,  # E8_merge_car_pessimism_similarity.py (optional)
    "RUN_REGRESSIONS": True,     # E9_regression_tfidf.py
    # Execution mode:
    "DRY_RUN": False,
}


@dataclass(frozen=True)
class Step:
    key: str
    script: str                 # path relative to scripts_dir
    deps: Tuple[str, ...] = ()


PIPELINE: List[Step] = [
    Step("tfidf_similarity", "E5_tfidf_cosine.py"),
    # assumes E8_merge_car_pessimism_similarity.py is in extension/ (parent of tfidf_cosine/)
    Step("merge_analysis_dataset", "../E8_merge_car_pessimism_similarity.py", deps=("tfidf_similarity",)),
    Step("regressions_tfidf", "E9_regression_tfidf.py", deps=("tfidf_similarity",)),
]


def get_project_root() -> Path:
    """
    Robustly find repo root even when called from extension/tfidf_cosine/.
    Strategy: walk up parents until we find expected root markers.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data_clean").exists() and (p / "outputs").exists():
            return p
    raise RuntimeError(
        "Could not locate project root. Expected to find 'data_clean/' and 'outputs/' in a parent directory."
    )


def expand_with_deps(wanted: Set[str], smap: Dict[str, Step], disabled: Set[str]) -> List[Step]:
    """Resolve dependencies for requested keys, exclude disabled, preserve PIPELINE order."""
    seen: Set[str] = set()

    def add(k: str) -> None:
        if k in disabled:
            return
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
    """
    Fail early with clear messages if upstream artifacts are missing.
    (We validate based on which steps are planned to run.)
    """
    keys = {s.key for s in steps}

    # Needed for TF-IDF similarity
    if "tfidf_similarity" in keys:
        require_file(
            project_root,
            "data_clean/ecb_statements_preprocessed.csv",
            "Run replication preprocess (step 4) to generate ecb_statements_preprocessed.csv.",
        )

    # Needed for merge dataset (optional)
    if "merge_analysis_dataset" in keys:
        require_file(
            project_root,
            "data_clean/ecb_pessimism_with_car.csv",
            "Run replication steps 6 + 7 to generate ecb_pessimism_with_car.csv.",
        )
        # similarity output will be produced by tfidf_similarity, so no need to require it here.

    # Needed for regressions
    if "regressions_tfidf" in keys:
        require_file(
            project_root,
            "data_clean/ecb_pessimism_with_car.csv",
            "Run replication steps 6 + 7 to generate ecb_pessimism_with_car.csv.",
        )
        require_file(
            project_root,
            "data_clean/controls_month_end.csv",
            "Run replication step 7b to generate controls_month_end.csv.",
        )
        # similarity output produced by tfidf_similarity (dependency)


def run_step(scripts_dir: Path, project_root: Path, step: Step, env: Dict[str, str], dry_run: bool) -> None:
    script_path = (scripts_dir / step.script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [sys.executable, str(script_path)]
    print(f"▶ {step.key:22s}  {script_path.relative_to(project_root) if project_root in script_path.parents else script_path}")

    if dry_run:
        print("  ", " ".join(cmd))
        return

    subprocess.run(cmd, check=True, cwd=str(project_root), env={**os.environ, **env})


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent          # .../extension/tfidf_cosine
    project_root = get_project_root()

    smap = {s.key: s for s in PIPELINE}
    wanted = set()

    if CONFIG["RUN_SIMILARITY"]:
        wanted.add("tfidf_similarity")
    if CONFIG["RUN_MERGE_DATASET"]:
        wanted.add("merge_analysis_dataset")
    if CONFIG["RUN_REGRESSIONS"]:
        wanted.add("regressions_tfidf")

    if not wanted:
        print("No steps selected (all RUN_* are False). Nothing to do.")
        return

    disabled: Set[str] = set()
    steps = expand_with_deps(wanted, smap, disabled)

    # Shared env window for all steps
    env = {
        "ECB_START_DATE": CONFIG["START_DATE"],
        "ECB_END_DATE": CONFIG["END_DATE"],
    }

    # Validate prerequisites (only those needed by selected steps)
    validate_prereqs(project_root, steps)

    # Execute in dependency order
    for step in steps:
        run_step(scripts_dir, project_root, step, env, CONFIG["DRY_RUN"])

    print("Done.")


if __name__ == "__main__":
    main()
