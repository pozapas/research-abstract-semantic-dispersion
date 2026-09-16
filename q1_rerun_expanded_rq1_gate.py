"""Run the guarded expanded-cohort RQ1 replication gate.

This script does not overwrite the locked Q1 manuscript outputs. It checks
whether the larger RQ2-expanded STEM sample preserves the descriptive RQ1
direction before the project uses that expanded sample in downstream causal
work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon, pdist
from scipy.stats import entropy, ttest_ind


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"

RNG_SEED = 20260424
CUTOFF_MONTH = pd.Timestamp("2022-12-01")
EXPECTED_ROWS = 425_728
EXPECTED_DIM = 384


@dataclass(frozen=True)
class ExpandedCohort:
    sample: pd.DataFrame
    embeddings: np.ndarray
    pre_idx: np.ndarray
    post_idx: np.ndarray


def ensure_dirs() -> None:
    RESULTS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)


def load_expanded_cohort() -> ExpandedCohort:
    sample_path = DATA / "q1_rq2_expanded_stem_sample.csv"
    embedding_path = DATA / "q1_rq2_expanded_embeddings.npy"
    manifest_path = RESULTS / "q1_rq2_expanded_embedding_manifest.json"

    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    if not embedding_path.exists():
        raise FileNotFoundError(embedding_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    usecols = [
        "id",
        "categories",
        "primary_category",
        "macro_field",
        "title",
        "abstract",
        "submission_month",
        "expanded_row_id",
    ]
    sample = pd.read_csv(sample_path, usecols=usecols, dtype={"id": str})
    sample = sample.sort_values("expanded_row_id").reset_index(drop=True)
    sample["submission_month"] = pd.to_datetime(sample["submission_month"], errors="coerce")

    row_ids = sample["expanded_row_id"].to_numpy(dtype=np.int64)
    expected_ids = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(row_ids, expected_ids):
        bad = int(np.where(row_ids != expected_ids)[0][0])
        raise ValueError(f"expanded_row_id is not contiguous at position {bad}: {row_ids[bad]}")

    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.shape != (EXPECTED_ROWS, EXPECTED_DIM):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    if len(sample) != embeddings.shape[0]:
        raise ValueError(f"Sample rows ({len(sample)}) and embeddings ({embeddings.shape[0]}) differ.")

    pre_idx = sample.index[sample["submission_month"] < CUTOFF_MONTH].to_numpy()
    post_idx = sample.index[sample["submission_month"] >= CUTOFF_MONTH].to_numpy()
    if len(pre_idx) == 0 or len(post_idx) == 0:
        raise ValueError("Pre/post split is empty.")

    return ExpandedCohort(sample=sample, embeddings=embeddings, pre_idx=pre_idx, post_idx=post_idx)


def _mean_pairwise_distance(embeddings: np.ndarray, indices: np.ndarray) -> float:
    values = np.asarray(embeddings[indices], dtype=np.float32)
    return float(np.mean(pdist(values, metric="euclidean")))


def bootstrap_embedding_distance(
    cohort: ExpandedCohort,
    n_boot: int = 60,
    sample_size: int = 1000,
) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for boot in range(n_boot):
        for period, pool in (("Pre-GPT", cohort.pre_idx), ("Post-GPT", cohort.post_idx)):
            chosen = rng.choice(pool, min(sample_size, len(pool)), replace=False)
            rows.append(
                {
                    "boot": boot,
                    "period": period,
                    "mean_distance": _mean_pairwise_distance(cohort.embeddings, chosen),
                    "sample_size": int(len(chosen)),
                }
            )
    draws = pd.DataFrame(rows)
    draws.to_csv(RESULTS / "q1_rq2_expanded_rq1_embedding_distance_bootstrap.csv", index=False)

    pre = draws.loc[draws["period"] == "Pre-GPT", "mean_distance"]
    post = draws.loc[draws["period"] == "Post-GPT", "mean_distance"]
    delta = float(post.mean() - pre.mean())
    summary = pd.DataFrame(
        [
            {
                "metric": "Mean pairwise SBERT distance",
                "pre_mean": float(pre.mean()),
                "pre_sd": float(pre.std(ddof=1)),
                "post_mean": float(post.mean()),
                "post_sd": float(post.std(ddof=1)),
                "delta": delta,
                "delta_pct": float(delta / pre.mean() * 100),
                "welch_p": float(ttest_ind(pre, post, equal_var=False).pvalue),
                "n_boot": int(n_boot),
                "sample_size": int(sample_size),
            }
        ]
    )
    summary.to_csv(RESULTS / "q1_rq2_expanded_rq1_embedding_distance_summary.csv", index=False)
    return summary


def monthly_embedding_distance(
    cohort: ExpandedCohort,
    sample_size: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.api as sm

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for month, group in cohort.sample.groupby("submission_month", sort=True):
        idx = group.index.to_numpy()
        if len(idx) < 15:
            continue
        chosen = rng.choice(idx, min(sample_size, len(idx)), replace=False)
        rows.append(
            {
                "month": month,
                "distance": _mean_pairwise_distance(cohort.embeddings, chosen),
                "n_papers": int(len(idx)),
                "sample_size": int(len(chosen)),
            }
        )

    ts = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    ts["time"] = np.arange(len(ts))
    ts["post_gpt"] = (ts["month"] >= CUTOFF_MONTH).astype(int)
    first_post = int(ts.loc[ts["post_gpt"] == 1, "time"].min())
    ts["time_after_gpt"] = np.maximum(0, ts["time"] - first_post)
    ts.to_csv(RESULTS / "q1_rq2_expanded_rq1_monthly_embedding_distance.csv", index=False)

    x = sm.add_constant(ts[["time", "post_gpt", "time_after_gpt"]])
    model = sm.OLS(ts["distance"], x).fit()
    hac = model.get_robustcov_results(cov_type="HAC", maxlags=3)
    coef = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "ols_se": model.bse.values,
            "ols_p": model.pvalues.values,
            "hac_se": hac.bse,
            "hac_p": hac.pvalues,
            "ci_low": hac.conf_int()[:, 0],
            "ci_high": hac.conf_int()[:, 1],
            "n_months": len(ts),
            "r_squared": model.rsquared,
        }
    )
    coef.to_csv(RESULTS / "q1_rq2_expanded_rq1_its_coefficients.csv", index=False)
    return ts, coef


def categorical_entropy(cohort: ExpandedCohort) -> pd.DataFrame:
    rows = []
    for column, label in (
        ("macro_field", "Macro-field composition entropy"),
        ("primary_category", "Primary arXiv-category composition entropy"),
    ):
        frame = cohort.sample[[column, "submission_month"]].dropna().copy()
        levels = sorted(frame[column].unique())
        pre = frame.loc[frame["submission_month"] < CUTOFF_MONTH, column]
        post = frame.loc[frame["submission_month"] >= CUTOFF_MONTH, column]

        def probs(series: pd.Series) -> np.ndarray:
            counts = series.value_counts().reindex(levels, fill_value=0).to_numpy(dtype=float)
            return counts / counts.sum()

        p_pre = probs(pre)
        p_post = probs(post)
        h_pre = float(entropy(p_pre))
        h_post = float(entropy(p_post))
        rows.append(
            {
                "metric": label,
                "column": column,
                "pre_n": int(len(pre)),
                "post_n": int(len(post)),
                "levels": int(len(levels)),
                "pre_entropy": h_pre,
                "post_entropy": h_post,
                "entropy_delta": h_post - h_pre,
                "entropy_delta_pct": (h_post - h_pre) / h_pre * 100 if h_pre else np.nan,
                "js_divergence": float(jensenshannon(p_pre, p_post) ** 2),
                "claim_boundary": "Composition diagnostic only; not a replacement for HDBSCAN topic entropy.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "q1_rq2_expanded_rq1_category_entropy.csv", index=False)
    return out


def bootstrap_tdi(
    cohort: ExpandedCohort,
    n_boot: int = 20,
    sample_size: int = 500,
) -> pd.DataFrame:
    try:
        from ripser import ripser
    except ImportError:
        out = pd.DataFrame(
            [
                {
                    "status": "blocked",
                    "reason": "ripser is not installed",
                    "n_boot": int(n_boot),
                    "sample_size": int(sample_size),
                }
            ]
        )
        out.to_csv(RESULTS / "q1_rq2_expanded_rq1_tdi_summary.csv", index=False)
        return out

    rng = np.random.default_rng(RNG_SEED)
    rows = []

    def tdi_value(indices: np.ndarray) -> float:
        values = np.asarray(cohort.embeddings[indices], dtype=np.float32)
        diagrams = ripser(values, maxdim=1)["dgms"]
        h1 = diagrams[1]
        if len(h1) == 0:
            return 0.0
        persistence = h1[:, 1] - h1[:, 0]
        persistence = persistence[np.isfinite(persistence)]
        return float(np.sum(persistence))

    for boot in range(n_boot):
        for period, pool in (("Pre-GPT", cohort.pre_idx), ("Post-GPT", cohort.post_idx)):
            chosen = rng.choice(pool, min(sample_size, len(pool)), replace=False)
            rows.append(
                {
                    "boot": boot,
                    "period": period,
                    "tdi_h1_persistence": tdi_value(chosen),
                    "sample_size": int(len(chosen)),
                }
            )
    draws = pd.DataFrame(rows)
    draws.to_csv(RESULTS / "q1_rq2_expanded_rq1_tdi_bootstrap.csv", index=False)

    pre = draws.loc[draws["period"] == "Pre-GPT", "tdi_h1_persistence"]
    post = draws.loc[draws["period"] == "Post-GPT", "tdi_h1_persistence"]
    delta = float(post.mean() - pre.mean())
    summary = pd.DataFrame(
        [
            {
                "metric": "TDI H1 persistence",
                "pre_mean": float(pre.mean()),
                "pre_sd": float(pre.std(ddof=1)),
                "post_mean": float(post.mean()),
                "post_sd": float(post.std(ddof=1)),
                "delta": delta,
                "delta_pct": float(delta / pre.mean() * 100) if pre.mean() else np.nan,
                "welch_p": float(ttest_ind(pre, post, equal_var=False).pvalue),
                "n_boot": int(n_boot),
                "sample_size": int(sample_size),
            }
        ]
    )
    summary.to_csv(RESULTS / "q1_rq2_expanded_rq1_tdi_summary.csv", index=False)
    return summary


def _load_locked_results() -> dict[str, float | str]:
    locked: dict[str, float | str] = {}
    distance_path = RESULTS / "q1_embedding_distance_summary.csv"
    tdi_path = RESULTS / "q1_tdi_summary.csv"
    topic_path = RESULTS / "q1_topic_entropy_amri.csv"

    if distance_path.exists():
        row = pd.read_csv(distance_path).iloc[0]
        locked["semantic_delta_pct"] = float(row["delta_pct"])
    if tdi_path.exists():
        row = pd.read_csv(tdi_path).iloc[0]
        if "delta_pct" in row:
            locked["tdi_delta_pct"] = float(row["delta_pct"])
    if topic_path.exists():
        topic = pd.read_csv(topic_path)
        row = topic.loc[topic["metric_group"] == "include_hdbscan_noise"].iloc[0]
        locked["topic_entropy_delta_pct"] = float(row["entropy_delta_pct"])
    return locked


def _direction_label(delta_pct: float | None) -> str:
    if delta_pct is None or pd.isna(delta_pct):
        return "unavailable"
    if delta_pct < 0:
        return "contraction"
    if delta_pct > 0:
        return "expansion"
    return "no change"


def write_gate_outputs(
    cohort: ExpandedCohort,
    distance: pd.DataFrame,
    monthly: pd.DataFrame,
    its: pd.DataFrame,
    category: pd.DataFrame,
    tdi: pd.DataFrame,
) -> dict[str, object]:
    locked = _load_locked_results()
    dist_row = distance.iloc[0].to_dict()
    tdi_available = "delta_pct" in tdi.columns and len(tdi) > 0 and not pd.isna(tdi.iloc[0].get("delta_pct", np.nan))
    tdi_row = tdi.iloc[0].to_dict() if tdi_available else {}

    semantic_pass = bool(dist_row["delta_pct"] < 0 and dist_row["welch_p"] < 0.05)
    tdi_pass = bool(tdi_available and tdi_row["delta_pct"] < 0 and tdi_row["welch_p"] < 0.05)
    if semantic_pass and tdi_pass:
        gate = "pass_for_descriptive_rq1_direction"
    elif semantic_pass and not tdi_available:
        gate = "partial_pass_semantic_only_tdi_blocked"
    elif semantic_pass:
        gate = "hold_semantic_pass_topology_not_confirmed"
    else:
        gate = "fail_semantic_direction_not_replicated"

    category_rows = category.set_index("column").to_dict(orient="index")
    manifest = {
        "script": "q1_rerun_expanded_rq1_gate.py",
        "rng_seed": RNG_SEED,
        "status": gate,
        "scope": "expanded guarded STEM sample for RQ2; does not overwrite locked Q1 outputs",
        "cutoff_policy": "monthly expanded sample uses post period starting 2022-12 because 2022-11 cannot isolate 30 Nov release day",
        "rows_total": int(len(cohort.sample)),
        "pre_rows": int(len(cohort.pre_idx)),
        "post_rows": int(len(cohort.post_idx)),
        "embedding_shape": [int(x) for x in cohort.embeddings.shape],
        "semantic_gate": {
            "pass": semantic_pass,
            "locked_delta_pct": locked.get("semantic_delta_pct"),
            "expanded_delta_pct": float(dist_row["delta_pct"]),
            "expanded_welch_p": float(dist_row["welch_p"]),
            "direction": _direction_label(float(dist_row["delta_pct"])),
        },
        "tdi_gate": {
            "pass": tdi_pass,
            "available": bool(tdi_available),
            "locked_delta_pct": locked.get("tdi_delta_pct"),
            "expanded_delta_pct": float(tdi_row["delta_pct"]) if tdi_available else None,
            "expanded_welch_p": float(tdi_row["welch_p"]) if tdi_available else None,
            "direction": _direction_label(float(tdi_row["delta_pct"])) if tdi_available else "blocked",
        },
        "category_composition": category_rows,
        "monthly_distance_rows": int(len(monthly)),
        "its_terms": its.to_dict(orient="records"),
        "claim_boundary": "Expanded sample may support RQ1 direction only after gate pass; causal RQ2 still requires adoption signal and monthly event-study rerun.",
        "outputs": {
            "embedding_distance_summary": "results/q1_rq2_expanded_rq1_embedding_distance_summary.csv",
            "monthly_distance": "results/q1_rq2_expanded_rq1_monthly_embedding_distance.csv",
            "its_coefficients": "results/q1_rq2_expanded_rq1_its_coefficients.csv",
            "category_entropy": "results/q1_rq2_expanded_rq1_category_entropy.csv",
            "tdi_summary": "results/q1_rq2_expanded_rq1_tdi_summary.csv",
            "table": "tables/q1_T14_expanded_rq1_replication_gate.tex",
        },
    }
    (RESULTS / "q1_rq2_expanded_rq1_gate_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    def sci(value: float | None, digits: int = 3) -> str:
        if value is None or pd.isna(value):
            return "--"
        if abs(value) < 0.001 and value != 0:
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"

    def p_text(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "--"
        if value < 0.001:
            return "$<0.001$"
        return f"{value:.3f}"

    tdi_decision = (
        rf"{sci(tdi_row['delta_pct'])}\%; {p_text(tdi_row['welch_p'])}; pass={str(tdi_pass).lower()}"
        if tdi_available
        else "blocked; ripser unavailable"
    )
    cat_primary = category_rows["primary_category"]
    cat_macro = category_rows["macro_field"]

    table = rf"""\begin{{table}}[t]
\centering
\caption{{Expanded-cohort RQ1 replication gate before using the larger RQ2 sample. The gate protects the manuscript from changing its descriptive foundation after adding data for causal identification.}}
\label{{tab:q1_expanded_rq1_replication_gate}}
\small
\setlength{{\tabcolsep}}{{4.2pt}}
\renewcommand{{\arraystretch}}{{1.18}}
\begin{{tabularx}}{{\linewidth}}{{>{{\raggedright\arraybackslash}}p{{2.7cm}} >{{\raggedright\arraybackslash}}X >{{\raggedleft\arraybackslash}}p{{2.3cm}} >{{\raggedright\arraybackslash}}p{{3.0cm}}}}
\toprule
\textbf{{Gate}} & \textbf{{Estimator and scope boundary}} & \textbf{{Expanded result}} & \textbf{{Decision}} \\
\midrule
\rowcolor{{q1teallight}}
Semantic geometry & Mean pairwise SBERT distance, {int(dist_row['n_boot'])} bootstraps of {int(dist_row['sample_size'])} papers per period. Locked direction: {_direction_label(locked.get('semantic_delta_pct'))}. & {sci(dist_row['delta_pct'])}\%; {p_text(dist_row['welch_p'])} & Direction replicated; pass={str(semantic_pass).lower()}. \\
Topological form & Vietoris--Rips $H_1$ persistence on landmark bootstrap samples. Locked direction: {_direction_label(locked.get('tdi_delta_pct'))}. & {tdi_decision} & Required before treating expanded data as full RQ1 replacement. \\
\rowcolor{{q1teallight}}
Monthly stability & Interrupted time-series over {len(monthly)} monthly cells using the expanded sample. & $R^2={sci(float(its['r_squared'].iloc[0]))}$ & Descriptive trend audit only; not causal. \\
Composition guard & Primary-category entropy; macro-field entropy is reported separately to detect scope drift. & Primary {sci(cat_primary['entropy_delta_pct'])}\%; macro {sci(cat_macro['entropy_delta_pct'])}\% & Composition diagnostic only; not HDBSCAN topic entropy. \\
\bottomrule
\end{{tabularx}}
\end{{table}}
"""
    (TABLES / "q1_T14_expanded_rq1_replication_gate.tex").write_text(table, encoding="utf-8")

    memo = f"""# Expanded RQ1 Replication Gate

Date: 2026-04-24

## Decision

`{gate}`

The expanded sample is a guarded RQ2 dataset. It should not overwrite the locked
Q1 manuscript outputs unless the gate is explicitly passed and the writing plan
states which outputs are being replaced.

## Key Results

- Rows: {len(cohort.sample):,} total; {len(cohort.pre_idx):,} pre; {len(cohort.post_idx):,} post.
- Semantic distance: {dist_row['delta_pct']:.3f}% change, Welch p = {dist_row['welch_p']:.3g}.
- TDI: {('%.3f%% change, Welch p = %.3g' % (tdi_row['delta_pct'], tdi_row['welch_p'])) if tdi_available else 'blocked because ripser is unavailable'}.
- Primary-category entropy: {cat_primary['entropy_delta_pct']:.3f}% change.
- Macro-field entropy: {cat_macro['entropy_delta_pct']:.3f}% change.

## Claim Boundary

This gate tests whether the larger STEM sample endangers RQ1 direction. It does
not answer RQ2. RQ2 still requires an adoption/style/disclosure exposure signal
and a monthly event-study rerun with pre-trend, placebo, and leave-field-out
diagnostics.
"""
    (RESULTS / "q1_rq2_expanded_rq1_gate_summary.txt").write_text(memo, encoding="utf-8")
    return manifest


def main() -> None:
    ensure_dirs()
    cohort = load_expanded_cohort()
    distance = bootstrap_embedding_distance(cohort)
    monthly, its = monthly_embedding_distance(cohort)
    category = categorical_entropy(cohort)
    tdi = bootstrap_tdi(cohort)
    manifest = write_gate_outputs(cohort, distance, monthly, its, category, tdi)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
