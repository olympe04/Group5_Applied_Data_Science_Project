# replication/main.py
# Run the full ECB text-analysis pipeline by executing each script in dependency order.
# I/O:
#   Inputs: project scripts + input files expected by each step (data_raw/, data_clean/).
#   Outputs: step-generated CSV/PNG files written under data_raw/, data_clean/, and outputs/.

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Pipeline configuration:
# - START_DATE / END_DATE are passed to downstream scripts via env vars (used for plots + regression windows).
# - RUN_SCRAPING toggles whether steps (1)-(2) are executed or skipped.
# - REQUIRE_RAW_IF_NO_SCRAPE enforces presence of step (2) output when scraping is disabled.
# - DRY_RUN prints commands instead of executing them.
CONFIG = {
    "START_DATE": "1999-01-01",
    "END_DATE": "2025-12-31",
    "RUN_SCRAPING": False,
    "REQUIRE_RAW_IF_NO_SCRAPE": True,
    "DRY_RUN": False,
}


@dataclass(frozen=True)
class Step:
    """Define one pipeline step with a unique key, a Python script name, and upstream dependency keys."""
    key: str
    script: str
    deps: Tuple[str, ...] = ()


# Ordered pipeline definition:
# - Order matters for execution and for dependency expansion.
# - deps must reference other Step.key values defined in this list.
PIPELINE: List[Step] = [
    Step("scrape_urls", "1_scraping_ecb.py"),
    Step("scrape_text", "2_scraping_statements.py", deps=("scrape_urls",)),
    Step("filter_raw", "3_filter_raw_before_preprocess.py", deps=("scrape_text",)),
    Step("preprocess", "4_pre-process.py", deps=("filter_raw",)),
    Step("similarity", "5_similarity_jaccard_bigrams.py", deps=("preprocess",)),
    Step("pessimism", "6_sentiment_pessimism_lm.py", deps=("preprocess",)),
    Step("controls", "7b_prepare_controls_month_end.py"),
    Step("event_study", "7_event_study_car.py", deps=("pessimism",)),
    Step("reg_table3", "8_regressions-table3.py", deps=("similarity", "controls")),
    Step("reg_table4", "8_regressions-table4.py", deps=("event_study", "similarity", "controls")),
    Step("summary_table2", "9_summary_table2.py", deps=("event_study", "similarity", "controls")),
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

        # Ensure all dependencies are scheduled before the step itself.
        for d in smap[k].deps:
            add(d)

        seen.add(k)

    for k in list(wanted):
        add(k)

    # Preserve the original pipeline order for execution.
    return [s for s in PIPELINE if s.key in seen]


def run_step(scripts_dir: Path, project_root: Path, step: Step, env: Dict[str, str], dry_run: bool) -> None:
    """
    Run one script as a subprocess with consistent working directory and injected environment variables.

    scripts_dir: folder containing the pipeline scripts (replication/)
    project_root: project root folder containing data_raw/, data_clean/, outputs/
    """
    script_path = scripts_dir / step.script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [sys.executable, str(script_path)]
    print(f"▶ {step.key:12s}  {step.script}")

    # DRY_RUN is useful for verifying ordering and commands without writing any outputs.
    if dry_run:
        print("  ", " ".join(cmd))
        return

    # Run as if we launched from the project root so relative paths like "data_raw/..." resolve correctly.
    subprocess.run(cmd, check=True, cwd=str(project_root), env={**os.environ, **env})


def main() -> None:
    """
    Compute which steps to run, validate prerequisites, then execute each script with a shared observation window.

    This version is shareable:
    - Scripts live in replication/
    - Data folders live at the project root (data_raw/, data_clean/, outputs/)
    - Running this file works from anywhere because subprocesses use cwd=project_root.
    """
    scripts_dir = Path(__file__).resolve().parent          # .../replication
    project_root = scripts_dir.parent                     # .../ (root of repo)

    smap = {s.key: s for s in PIPELINE}

    # Default behavior: attempt to run the entire pipeline (disabled steps are removed below).
    wanted = {s.key for s in PIPELINE}

    # Step selection logic
    disabled: Set[str] = set()

    # Toggle scraping steps (1)-(2) as a single switch.
    if not CONFIG["RUN_SCRAPING"]:
        disabled |= {"scrape_urls", "scrape_text"}

        # If scraping is skipped, step (3) still needs the raw file produced by step (2).
        if CONFIG.get("REQUIRE_RAW_IF_NO_SCRAPE", True):
            raw_path = project_root / "data_raw" / "ecb_statements_raw.csv"
            if not raw_path.exists():
                raise FileNotFoundError(
                    f"RUN_SCRAPING=False but missing required input for step 3:\n"
                    f"  {raw_path}\n"
                    "Either set RUN_SCRAPING=True, or provide the CSV produced by step 2."
                )

    # Resolve dependencies for the final execution set.
    steps = expand_with_deps(wanted, smap, disabled)

    # Shared environment passed to downstream scripts
    env = {
        "ECB_START_DATE": CONFIG["START_DATE"],
        "ECB_END_DATE": CONFIG["END_DATE"],
    }

    # Print which steps are skipped
    if disabled:
        print(f"Skipping steps: {', '.join(sorted(disabled))}")

    # Execute steps sequentially (dependency order is guaranteed by expand_with_deps)
    for step in steps:
        run_step(scripts_dir, project_root, step, env, CONFIG["DRY_RUN"])

    print("Done.")


if __name__ == "__main__":
    main()
