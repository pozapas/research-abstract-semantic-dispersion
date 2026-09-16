"""Recompute the exploratory topic lens with declared, reproducible settings.

The pooled locked cohort is projected to 50 principal components and clustered
once.  A small HDBSCAN grid shows how the entropy direction and noise share
change with the two main density parameters.  Cluster labels therefore have a
common support in the pre- and post-cutoff periods.
"""

from __future__ import annotations

import json
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CUTOFF = pd.Timestamp("2022-11-30")
RNG_SEED = 20260424
SETTINGS = ((50, 5), (100, 10), (200, 20))


def summarize(labels: np.ndarray, post: np.ndarray, policy: str) -> dict[str, float | int | str]:
    keep = np.ones(len(labels), dtype=bool) if policy == "include_noise" else labels != -1
    used_labels = labels[keep]
    used_post = post[keep]
    support = np.sort(np.unique(used_labels))

    def probabilities(period: int) -> np.ndarray:
        values, counts = np.unique(used_labels[used_post == period], return_counts=True)
        mapping = dict(zip(values.tolist(), counts.tolist()))
        result = np.array([mapping.get(label, 0) for label in support], dtype=float)
        return result / result.sum()

    pre = probabilities(0)
    pos = probabilities(1)
    pre_entropy = float(entropy(pre))
    post_entropy = float(entropy(pos))
    return {
        "noise_policy": policy,
        "n_used": int(keep.sum()),
        "n_clusters_on_support": int(len(support)),
        "pre_entropy": pre_entropy,
        "post_entropy": post_entropy,
        "entropy_delta": post_entropy - pre_entropy,
        "entropy_delta_pct": 100.0 * (post_entropy - pre_entropy) / pre_entropy,
        "js_divergence": float(jensenshannon(pre, pos) ** 2),
    }


def main() -> None:
    cohort = pd.read_csv(
        DATA / "q1_arxiv_analysis_cohort.csv",
        usecols=["row_id", "analysis_date"],
    )
    cohort["analysis_date"] = pd.to_datetime(cohort["analysis_date"], errors="raise")
    post = (cohort["analysis_date"] >= CUTOFF).to_numpy(dtype=np.int8)
    embeddings = np.load(DATA / "q1_locked_title_abstract_embeddings.npy", mmap_mode="r")
    matrix = np.asarray(embeddings, dtype=np.float32)

    pca = PCA(n_components=50, svd_solver="randomized", random_state=RNG_SEED)
    reduced = pca.fit_transform(matrix).astype(np.float32, copy=False)

    rows: list[dict[str, float | int | str]] = []
    primary_labels: np.ndarray | None = None
    for min_cluster_size, min_samples in SETTINGS:
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False,
            core_dist_n_jobs=-1,
        )
        labels = model.fit_predict(reduced)
        if (min_cluster_size, min_samples) == (100, 10):
            primary_labels = labels.copy()
        base = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "n_nonnoise_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "noise_share": float(np.mean(labels == -1)),
        }
        for policy in ("include_noise", "exclude_noise"):
            rows.append(base | summarize(labels, post, policy))

    pd.DataFrame(rows).to_csv(RESULTS / "q1_topic_stability.csv", index=False)
    if primary_labels is not None:
        pd.DataFrame(
            {
                "row_id": cohort.row_id.to_numpy(dtype=np.int64),
                "topic_cluster": primary_labels,
            }
        ).to_csv(RESULTS / "q1_topic_clusters_locked.csv", index=False)

    manifest = {
        "script": Path(__file__).name,
        "fit_sample": "pooled locked cohort, one clustering for both periods",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "text_input": "title + '. ' + abstract",
        "preprocessing": "50-component randomized PCA on unit-normalized 384-dimensional embeddings",
        "pca_random_state": RNG_SEED,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "hdbscan_metric": "euclidean",
        "cluster_selection_method": "eom",
        "settings": [
            {"min_cluster_size": size, "min_samples": samples}
            for size, samples in SETTINGS
        ],
        "primary_setting": {"min_cluster_size": 100, "min_samples": 10},
        "outputs": [
            "results/q1_topic_stability.csv",
            "results/q1_topic_clusters_locked.csv",
        ],
    }
    (RESULTS / "q1_topic_stability_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
