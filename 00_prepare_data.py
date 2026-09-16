"""Build the extracted arXiv metadata and the fixed locked cohort.

The script reproduces the original two-stage locked-cohort rule:
1. Keep arXiv records with an eligible STEM category.
2. Draw 50,000 source rows with the pandas-compatible seed 42.
3. Use the first-version date and keep papers from 2018 through 2025.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PREFIXES = ("cs.", "math.", "stat.", "physics.", "q-bio.", "eess.")
SAMPLE_SEED = 42
SAMPLE_SIZE = 50_000
START = pd.Timestamp("2018-01-01")
END = pd.Timestamp("2025-12-31")
CUTOFF = pd.Timestamp("2022-11-30")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=DATA / "archive" / "arxiv-metadata-oai-snapshot.json",
    )
    parser.add_argument(
        "--reuse-extracted",
        action="store_true",
        help="Reuse data/arxiv_meta.csv and only rebuild the sample and cohort.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_version_date(record: dict) -> str:
    versions = record.get("versions") or []
    if not versions:
        return ""
    value = versions[0].get("created", "")
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return parsed.tz_convert(None).date().isoformat()


def extract(raw_path: Path, output_path: Path) -> int:
    count = 0
    with raw_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target)
        writer.writerow(
            ["id", "categories", "title", "abstract", "update_date", "submitted_date"]
        )
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            categories = str(record.get("categories", ""))
            if not any(item.startswith(PREFIXES) for item in categories.split()):
                continue
            writer.writerow(
                [
                    record.get("id", ""),
                    categories,
                    str(record.get("title", "")).replace("\n", " ").strip(),
                    str(record.get("abstract", "")).replace("\n", " ").strip(),
                    record.get("update_date", ""),
                    first_version_date(record),
                ]
            )
            count += 1
    return count


def sample_rows(metadata_path: Path, sample_path: Path) -> pd.DataFrame:
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        total = sum(1 for _ in handle) - 1
    size = min(SAMPLE_SIZE, total)
    chosen = np.random.RandomState(SAMPLE_SEED).choice(total, size=size, replace=False)
    rank = {int(source_row): int(order) for order, source_row in enumerate(chosen)}

    pieces = []
    offset = 0
    for chunk in pd.read_csv(metadata_path, dtype={"id": str}, chunksize=100_000):
        positions = np.arange(offset, offset + len(chunk), dtype=np.int64)
        keep = np.fromiter((int(value) in rank for value in positions), dtype=bool)
        if keep.any():
            part = chunk.loc[keep].copy()
            selected_positions = positions[keep]
            part["_sample_order"] = [rank[int(value)] for value in selected_positions]
            pieces.append(part)
        offset += len(chunk)

    sample = pd.concat(pieces, ignore_index=True).sort_values("_sample_order")
    sample = sample.drop(columns="_sample_order").reset_index(drop=True)
    sample.to_csv(sample_path, index=False)
    return sample


def build_cohort(sample: pd.DataFrame) -> pd.DataFrame:
    sample = sample.copy()
    sample["submitted_date"] = pd.to_datetime(sample["submitted_date"], errors="coerce")
    sample["update_date"] = pd.to_datetime(sample["update_date"], errors="coerce")
    sample["analysis_date"] = sample["submitted_date"].fillna(sample["update_date"])
    sample["row_id"] = np.arange(len(sample), dtype=np.int64)
    cohort = sample.loc[sample.analysis_date.between(START, END)].copy()
    columns = [
        "id",
        "categories",
        "title",
        "analysis_date",
        "submitted_date",
        "update_date",
        "row_id",
    ]
    return cohort[columns]


def main() -> None:
    args = arguments()
    raw_path = args.raw.resolve()
    metadata_path = DATA / "arxiv_meta.csv"
    sample_path = DATA / "arxiv_meta_sampled.csv"
    cohort_path = DATA / "q1_arxiv_analysis_cohort.csv"
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    if not args.reuse_extracted:
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        extracted_rows = extract(raw_path, metadata_path)
    else:
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            extracted_rows = sum(1 for _ in handle) - 1

    sample = sample_rows(metadata_path, sample_path)
    cohort = build_cohort(sample)
    cohort.to_csv(cohort_path, index=False)
    pre = int((cohort.analysis_date < CUTOFF).sum())
    post = int((cohort.analysis_date >= CUTOFF).sum())

    manifest = {
        "script": Path(__file__).name,
        "source": str(raw_path),
        "source_sha256": file_sha256(raw_path) if raw_path.exists() else None,
        "eligible_rows": extracted_rows,
        "sample_seed": SAMPLE_SEED,
        "sample_rows": len(sample),
        "cohort_start": START.date().isoformat(),
        "cohort_end": END.date().isoformat(),
        "cutoff": CUTOFF.date().isoformat(),
        "cohort_rows": len(cohort),
        "pre_rows": pre,
        "post_rows": post,
    }
    (RESULTS / "q1_cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
