"""Check the tracked release and its headline numerical results."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{actual} != {expected}")


def check_numbers() -> None:
    exact = pd.read_csv(RESULTS / "q1_exact_semantic_distance.csv").set_index("period")
    close(float(exact.loc["Pre-GPT", "mean_pairwise_distance"]), 1.3533809061279731)
    close(float(exact.loc["Post-GPT", "mean_pairwise_distance"]), 1.331599760170067)

    decomp = pd.read_csv(
        RESULTS / "q1_rq2_decomposition_summary.csv", index_col=0
    ).iloc[:, 0]
    close(float(decomp["dS_full"]), -0.007698716746973755)
    close(float(decomp["leaveout_change"]), -0.005350914790499983)
    close(float(decomp["composition_gap_change"]), -0.0023478019564737718)
    close(float(decomp["identity_max_resid"]), 4.440892098500626e-16, 1e-20)


def check_no_secrets() -> None:
    patterns = [
        re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]+"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    ]
    excluded = {".git"}
    suffixes = {".py", ".md", ".tex", ".json", ".yml", ".yaml", ".txt", ".csv"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path.suffix.lower() not in suffixes or path.stat().st_size > 5_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                raise AssertionError(f"Possible secret in {path.relative_to(ROOT)}")


def check_required_files() -> None:
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "requirements.txt",
        "data/source_manifest.json",
        "results/q1_exact_semantic_distance.csv",
        "results/q1_rq2_decomposition_summary.csv",
    ]
    missing = [item for item in required if not (ROOT / item).exists()]
    if missing:
        raise AssertionError(f"Missing tracked files: {missing}")


def main() -> None:
    check_required_files()
    check_numbers()
    check_no_secrets()
    manifest = json.loads((ROOT / "data" / "source_manifest.json").read_text())
    if len(manifest["local_snapshot_sha256"]) != 64:
        raise AssertionError("Invalid source SHA-256 value")
    print("Release checks passed.")


if __name__ == "__main__":
    main()
