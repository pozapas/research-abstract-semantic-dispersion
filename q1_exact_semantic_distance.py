"""Compute exact locked-cohort semantic-dispersion statistics.

The earlier pipeline averaged 50 statistics from random samples of 1,000
abstracts.  This script computes the mean Euclidean distance over every distinct
pair in each period.  It uses SciPy's condensed distance representation and
therefore needs about 1.1 GB of temporary memory for the larger period.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CUTOFF = pd.Timestamp("2022-11-30")


def main() -> None:
    cohort = pd.read_csv(
        DATA / "q1_arxiv_analysis_cohort.csv",
        usecols=["row_id", "analysis_date"],
    )
    cohort["analysis_date"] = pd.to_datetime(cohort["analysis_date"], errors="raise")
    embeddings = np.load(DATA / "q1_locked_title_abstract_embeddings.npy", mmap_mode="r")

    rows: list[dict[str, float | int | str]] = []
    for period, mask in (
        ("Pre-GPT", cohort["analysis_date"] < CUTOFF),
        ("Post-GPT", cohort["analysis_date"] >= CUTOFF),
    ):
        indices = cohort.index[mask].to_numpy(dtype=np.int64)
        values = pdist(np.asarray(embeddings[indices], dtype=np.float32), metric="euclidean")
        rows.append(
            {
                "period": period,
                "n_abstracts": int(len(indices)),
                "n_distinct_pairs": int(len(values)),
                "mean_pairwise_distance": float(values.mean()),
                "distance_sd": float(values.std(ddof=1)),
            }
        )
        del values

    output = pd.DataFrame(rows)
    pre = float(output.loc[output.period == "Pre-GPT", "mean_pairwise_distance"].iloc[0])
    post = float(output.loc[output.period == "Post-GPT", "mean_pairwise_distance"].iloc[0])
    output["delta_post_minus_pre"] = post - pre
    output["delta_pct_of_pre"] = 100.0 * (post - pre) / pre
    output.to_csv(RESULTS / "q1_exact_semantic_distance.csv", index=False)

    manifest = {
        "script": Path(__file__).name,
        "cutoff": str(CUTOFF.date()),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "text_input": "title + '. ' + abstract",
        "model_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "distance": "Euclidean distance on unit-normalized sentence embeddings",
        "estimand": "finite-corpus mean over all distinct unordered abstract pairs within period",
        "output": "results/q1_exact_semantic_distance.csv",
    }
    (RESULTS / "q1_exact_semantic_distance_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
