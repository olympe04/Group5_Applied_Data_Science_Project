# extension/main.py
# Run the extension pipeline (TF-IDF similarity + merged dataset + TF-IDF regressions/tables).
# This main is designed to live in extension/ and execute extension scripts while
# reading/writing datasets under the project root folders:
#   data_clean/, data_features/, outputs/
#
# Usage: python extension/main.py
# (Optional) set CONFIG below to enable/disable steps.

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


CONFIG = {
    # Shared window passed to scripts that read ECB_START_DATE / ECB_END_DATE
    "START_DATE": "1999-01-01",
    "END_DATE": "2013-12-31",

    # Control which steps run
    "RUN_TFIDF_SIM": True,
    "RUN_MERGE": True,
    "RUN_TFIDF_TABLES": True,

    # Safety: if a step is disabled, you can still require its output to exist
    "REQUIRE_TFIDF_IF_SKIPPED": True,
    "REQUIRE_MERGE_IF_SKIPPED": False,

    # Debug
    "DRY_RUN": False,
}


@dataclass(frozen=True)
class Step:
    key: str
    script: str
    deps: Tuple[str, ...] = ()


PIPELINE: List[Step] = [
    Step("tfidf_similarity", "E5_tfidf_cosine.py"),
    Step("merge_dataset", "E8_merge_car_pessimism_similarity.py", deps=("tfidf_similarity",)),
    Step("tfidf_tables", "E9_regression_tfidf.py", deps=("tfidf_similarity",)),
]


def expand_with_deps(wanted: Set[str], smap: Dict[str, Step], disabled: Set[str]) -> List[Step]:
    """Resolve dependencies for the requested step keys, excluding disabled steps, and return steps in PIPELINE order."""
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


def run_step(scripts_dir: Path, project_root: Path, step: Step, env: Dict[str, str], dry_run: bool) -> None:
    """Run one extension script as a subprocess with cwd=project_root and injected env vars."""
    script_path = scripts_dir / step.script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [sys.executable, str(script_path)]
    print(f"▶ {step.key:16s}  {step.script}")

    if dry_run:
        print("  ", " ".join(cmd))
        return

    subprocess.run(cmd, check=True, cwd=str(project_root), env={**os.environ, **env})


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent     # .../extension
    project_root = scripts_dir.parent                 # .../ (repo root)

    smap = {s.key: s for s in PIPELINE}
    wanted = {s.key for s in PIPELINE}
    disabled: Set[str] = set()

    # Toggle steps via CONFIG
    if not CONFIG.get("RUN_TFIDF_SIM", True):
        disabled.add("tfidf_similarity")
        if CONFIG.get("REQUIRE_TFIDF_IF_SKIPPED", True):
            tfidf_out = project_root / "data_features" / "ecb_similarity_tfidf.csv"
            if not tfidf_out.exists():
                raise FileNotFoundError(
                    "RUN_TFIDF_SIM=False but missing required TF-IDF output:\n"
                    f"  {tfidf_out}\n"
                    "Either set RUN_TFIDF_SIM=True, or provide the CSV produced by the TF-IDF step."
                )

    if not CONFIG.get("RUN_MERGE", True):
        disabled.add("merge_dataset")
        if CONFIG.get("REQUIRE_MERGE_IF_SKIPPED", False):
            merged_out = project_root / "data_clean" / "ecb_analysis_dataset.csv"
            if not merged_out.exists():
                raise FileNotFoundError(
                    "RUN_MERGE=False but missing required merged output:\n"
                    f"  {merged_out}\n"
                    "Either set RUN_MERGE=True, or provide the merged dataset file."
                )

    if not CONFIG.get("RUN_TFIDF_TABLES", True):
        disabled.add("tfidf_tables")

    # Resolve dependency-expanded execution set
    steps = expand_with_deps(wanted, smap, disabled)

    env = {
        "ECB_START_DATE": CONFIG["START_DATE"],
        "ECB_END_DATE": CONFIG["END_DATE"],
    }

    if disabled:
        print(f"Skipping steps: {', '.join(sorted(disabled))}")

    for step in steps:
        run_step(scripts_dir, project_root, step, env, CONFIG["DRY_RUN"])

    print("Done.")


if __name__ == "__main__":
    main()
