"""Run the reproducible analysis in dependency order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / script), *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "locked", "expanded", "figures", "verify", "all"),
    )
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.stage in {"prepare", "all"}:
        prepare_args = ["--raw", str(args.raw)] if args.raw else []
        run("00_prepare_data.py", *prepare_args)

    if args.stage in {"locked", "all"}:
        run("q1_embed_locked_cohort.py", "--batch-size", str(args.batch_size))
        run("q1_exact_semantic_distance.py")
        run("q1_locked_diagnostics.py")
        run("q1_topic_stability.py")

    if args.stage in {"expanded", "all"}:
        run("q1_prepare_expanded_rq2_cohort.py")
        run("q1_embed_expanded_rq2_sample.py", "--batch-size", str(args.batch_size))
        run("q1_build_expanded_rq2_event_study.py")
        run("q1_rq2_phase1_inference.py")
        run("q1_rq2_phase2_leaveout_dose.py")
        run("q1_rq2_phase3_shiftshare.py")
        run("q1_rq2_phase4_synthesis.py")
        run("q1_rq2_decomposition_verify.py")
        run("q1_rq1_standardized.py")

    if args.stage in {"figures", "all"}:
        run("q1_make_figures.py")

    if args.stage in {"verify", "all"}:
        run("verify_release.py")


if __name__ == "__main__":
    main()
