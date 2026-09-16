"""Recompute locked-cohort diagnostic resampling and monthly estimates.

The primary finite-corpus result is produced by q1_exact_semantic_distance.py.
The repeated subsamples in this script are diagnostics for visualization and
method sensitivity.  They are not confidence intervals for the full corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from ripser import ripser
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RNG_SEED = 20260424
CUTOFF = pd.Timestamp("2022-11-30")


def mean_distance(matrix: np.ndarray, indices: np.ndarray) -> float:
    return float(np.mean(pdist(np.asarray(matrix[indices], dtype=np.float32))))


def repeated_semantic_subsamples(
    matrix: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    n_draws: int = 100,
    sample_size: int = 1000,
) -> None:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for boot in range(n_draws):
        for period, pool in (("Pre-GPT", pre), ("Post-GPT", post)):
            chosen = rng.choice(pool, size=sample_size, replace=False)
            rows.append(
                {
                    "boot": boot,
                    "period": period,
                    "mean_distance": mean_distance(matrix, chosen),
                    "sample_size": sample_size,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "q1_embedding_distance_bootstrap.csv", index=False)

    exact = pd.read_csv(RESULTS / "q1_exact_semantic_distance.csv")
    exact = exact.set_index("period")
    pre_draw = out.loc[out.period == "Pre-GPT", "mean_distance"]
    post_draw = out.loc[out.period == "Post-GPT", "mean_distance"]
    pre_exact = float(exact.loc["Pre-GPT", "mean_pairwise_distance"])
    post_exact = float(exact.loc["Post-GPT", "mean_pairwise_distance"])
    summary = pd.DataFrame(
        [
            {
                "metric": "Mean pairwise SBERT distance",
                "pre_mean": pre_exact,
                "pre_sd": float(pre_draw.std(ddof=1)),
                "post_mean": post_exact,
                "post_sd": float(post_draw.std(ddof=1)),
                "delta": post_exact - pre_exact,
                "delta_pct": 100.0 * (post_exact - pre_exact) / pre_exact,
                "n_draws": n_draws,
                "subsample_size": sample_size,
                "interval_role": "empirical repeated-subsample diagnostic; not a corpus confidence interval",
            }
        ]
    )
    summary.to_csv(RESULTS / "q1_embedding_distance_summary.csv", index=False)


def repeated_tdi_subsamples(
    matrix: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    n_draws: int = 20,
    sample_size: int = 500,
) -> None:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for boot in range(n_draws):
        for period, pool in (("Pre-GPT", pre), ("Post-GPT", post)):
            chosen = rng.choice(pool, size=sample_size, replace=False)
            diagram = ripser(np.asarray(matrix[chosen], dtype=np.float32), maxdim=1)["dgms"][1]
            persistence = diagram[:, 1] - diagram[:, 0] if len(diagram) else np.array([])
            persistence = persistence[np.isfinite(persistence)]
            rows.append(
                {
                    "boot": boot,
                    "period": period,
                    "tdi_h1_persistence": float(persistence.sum()),
                    "sample_size": sample_size,
                    "maxdim": 1,
                    "coefficient_field": 2,
                    "infinite_bars": "excluded",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "q1_tdi_bootstrap.csv", index=False)

    values = {}
    for period, group in out.groupby("period"):
        x = group.tdi_h1_persistence.to_numpy()
        values[period] = {
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)),
            "q025": float(np.quantile(x, 0.025)),
            "q975": float(np.quantile(x, 0.975)),
        }
    pre_mean = values["Pre-GPT"]["mean"]
    post_mean = values["Post-GPT"]["mean"]
    summary = pd.DataFrame(
        [
            {
                "metric": "TDI H1 total persistence",
                "pre_mean": pre_mean,
                "pre_sd": values["Pre-GPT"]["sd"],
                "pre_q025": values["Pre-GPT"]["q025"],
                "pre_q975": values["Pre-GPT"]["q975"],
                "post_mean": post_mean,
                "post_sd": values["Post-GPT"]["sd"],
                "post_q025": values["Post-GPT"]["q025"],
                "post_q975": values["Post-GPT"]["q975"],
                "delta": post_mean - pre_mean,
                "delta_pct": 100.0 * (post_mean - pre_mean) / pre_mean,
                "n_draws": n_draws,
                "subsample_size": sample_size,
                "interval_role": "empirical repeated-subsample diagnostic; marginal intervals may overlap",
            }
        ]
    )
    summary.to_csv(RESULTS / "q1_tdi_summary.csv", index=False)


def monthly_its(matrix: np.ndarray, cohort: pd.DataFrame, sample_size: int = 500) -> None:
    rng = np.random.default_rng(RNG_SEED)
    cohort = cohort.copy()
    cohort["month"] = cohort.analysis_date.dt.to_period("M")
    rows = []
    for month, group in cohort.groupby("month", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        if len(indices) < 15:
            continue
        chosen = rng.choice(indices, size=min(sample_size, len(indices)), replace=False)
        rows.append(
            {
                "month": month.to_timestamp(),
                "distance": mean_distance(matrix, chosen),
                "n_papers": int(len(indices)),
                "sample_size": int(len(chosen)),
            }
        )
    monthly = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    monthly["time"] = np.arange(len(monthly))
    monthly["post_gpt"] = (monthly.month >= CUTOFF).astype(int)
    first_post = int(monthly.loc[monthly.post_gpt == 1, "time"].min())
    monthly["time_after_gpt"] = np.maximum(0, monthly.time - first_post)
    monthly.to_csv(RESULTS / "q1_monthly_embedding_distance.csv", index=False)

    design = sm.add_constant(monthly[["time", "post_gpt", "time_after_gpt"]])
    fit = sm.OLS(monthly.distance, design).fit()
    hac = fit.get_robustcov_results(cov_type="HAC", maxlags=3)
    coefficients = pd.DataFrame(
        {
            "term": fit.params.index,
            "coef": fit.params.to_numpy(),
            "ols_se": fit.bse.to_numpy(),
            "ols_p": fit.pvalues.to_numpy(),
            "hac_se": hac.bse,
            "hac_p": hac.pvalues,
            "ci_low": hac.conf_int()[:, 0],
            "ci_high": hac.conf_int()[:, 1],
            "n_months": len(monthly),
            "r_squared": fit.rsquared,
        }
    )
    coefficients.to_csv(RESULTS / "q1_its_embedding_distance_coefficients.csv", index=False)


def idea_space_projection(matrix: np.ndarray, cohort: pd.DataFrame, sample_size: int = 2500) -> None:
    """Create a balanced display sample and fit one pooled PCA projection."""
    rng = np.random.default_rng(RNG_SEED)
    frame = cohort.copy()
    frame["period"] = np.where(frame.analysis_date < CUTOFF, "Pre-GPT", "Post-GPT")
    parts = []
    for period, group in frame.groupby("period", sort=False):
        chosen = rng.choice(group.index.to_numpy(), size=min(sample_size, len(group)), replace=False)
        parts.append(frame.loc[chosen, ["id", "analysis_date", "period"]])
    display = pd.concat(parts).sort_index()
    projected = PCA(n_components=2, random_state=RNG_SEED).fit_transform(
        np.asarray(matrix[display.index.to_numpy(dtype=np.int64)], dtype=np.float32)
    )
    display["pc1"] = projected[:, 0]
    display["pc2"] = projected[:, 1]
    display.to_csv(RESULTS / "q1_idea_space_projection.csv", index=False)


def main() -> None:
    cohort = pd.read_csv(DATA / "q1_arxiv_analysis_cohort.csv")
    cohort["analysis_date"] = pd.to_datetime(cohort.analysis_date, errors="raise")
    matrix = np.load(DATA / "q1_locked_title_abstract_embeddings.npy", mmap_mode="r")
    pre = cohort.index[cohort.analysis_date < CUTOFF].to_numpy(dtype=np.int64)
    post = cohort.index[cohort.analysis_date >= CUTOFF].to_numpy(dtype=np.int64)

    repeated_semantic_subsamples(matrix, pre, post)
    repeated_tdi_subsamples(matrix, pre, post)
    monthly_its(matrix, cohort)
    idea_space_projection(matrix, cohort)
    manifest = {
        "script": Path(__file__).name,
        "rng_seed": RNG_SEED,
        "semantic_subsamples": {"draws_per_period": 100, "size": 1000},
        "tdi_subsamples": {
            "draws_per_period": 20,
            "landmarks": 500,
            "complex": "Vietoris-Rips",
            "homology_dimension": 1,
            "coefficient_field": 2,
            "filtration": "ripser default; maximum edge equals the sample diameter",
            "infinite_bars": "excluded",
            "scale_normalization": "none; fixed landmark count and unit-normalized embedding scale",
        },
        "monthly_its": {"subsample_cap": 500, "hac_lags": 3},
    }
    (RESULTS / "q1_locked_diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
