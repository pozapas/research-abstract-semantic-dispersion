"""Fit the category-trend model and build a neutral diagnostic summary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def fit_with_trends(
    frame: pd.DataFrame,
    exposure: str,
    outcome: str,
    control: str,
    weights: str,
) -> dict[str, float]:
    data = frame.loc[frame[outcome].notna()].copy().reset_index(drop=True)
    data["did_term"] = data[exposure].astype(float) * data["post"].astype(float)
    field_fe = pd.get_dummies(
        data.primary_category, prefix="field", drop_first=True, dtype=float
    )
    month_fe = pd.get_dummies(
        data.month.astype(str), prefix="month", drop_first=True, dtype=float
    )
    event_time = data.event_time.astype(float).to_numpy()
    field_trends = pd.DataFrame(
        field_fe.to_numpy() * event_time[:, None],
        columns=[f"trend_{column}" for column in field_fe.columns],
    )
    design = pd.concat(
        [
            data[["did_term", control]].reset_index(drop=True),
            field_fe.reset_index(drop=True),
            month_fe.reset_index(drop=True),
            field_trends.reset_index(drop=True),
        ],
        axis=1,
    )
    design = design.loc[:, design.nunique() > 1]
    design.insert(0, "const", 1.0)
    model = sm.WLS(
        data[outcome].astype(float),
        design.astype(float),
        weights=data[weights].astype(float).clip(lower=1.0),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": data.primary_category.astype(str)},
    )
    interval = model.conf_int().loc["did_term"]
    return {
        "coef": float(model.params["did_term"]),
        "se": float(model.bse["did_term"]),
        "p_cat": float(model.pvalues["did_term"]),
        "ci_low": float(interval.iloc[0]),
        "ci_high": float(interval.iloc[1]),
        "n_cells": int(len(data)),
        "n_categories": int(data.primary_category.nunique()),
    }


def label_permutation(
    frame: pd.DataFrame,
    draws: int = 2_000,
    seed: int = 20260630,
) -> tuple[dict[str, float], float]:
    observed = fit_with_trends(
        frame, "cat_adoption_z", "dist_leaveout", "log_n_non", "n_non"
    )
    field_exposure = frame.groupby("primary_category").cat_adoption_z.first()
    categories = field_exposure.index.to_numpy()
    values = field_exposure.to_numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        mapping = dict(zip(categories, rng.permutation(values)))
        permuted = frame.copy()
        permuted["cat_adoption_z"] = permuted.primary_category.map(mapping)
        estimates.append(
            fit_with_trends(
                permuted,
                "cat_adoption_z",
                "dist_leaveout",
                "log_n_non",
                "n_non",
            )["coef"]
        )
    estimates = np.asarray(estimates)
    p_value = (
        np.sum(np.abs(estimates) >= abs(observed["coef"])) + 1
    ) / (len(estimates) + 1)
    return observed, float(p_value)


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    panel = pd.read_csv(RESULTS / "q1_rq2_phase2_panel.csv")
    eligible = panel[
        (panel.n_non >= 15)
        & panel.dist_leaveout.notna()
        & panel.cat_adoption.notna()
    ].copy()
    counts = eligible.groupby("primary_category").post.agg(
        pre=lambda values: int((values == 0).sum()),
        post=lambda values: int((values == 1).sum()),
    )
    categories = counts[(counts.pre >= 24) & (counts.post >= 18)].index
    eligible = eligible[eligible.primary_category.isin(categories)].copy()

    trend_result, trend_permutation_p = label_permutation(eligible)
    phase1 = load_json("q1_rq2_phase1_manifest.json")
    phase2 = load_json("q1_rq2_phase2_manifest.json")
    phase3 = load_json("q1_rq2_phase3_manifest.json")
    p2 = phase2["results"][0]
    p3 = phase3["results"][0]

    rows = [
        {
            "design": "P1 Nature adoption",
            "gradient": "external survey",
            "outcome": "full",
            "beta": phase1["baseline"]["coef"],
            "label_permutation_p": phase1["ri"]["p_ri_two"],
            "pretrend_p": 0.318,
            "drop_cs": "wide interval includes zero",
            "note": "reference-model dependent",
        },
        {
            "design": "P2 Marker dose",
            "gradient": "category marker share",
            "outcome": "leave-out",
            "beta": p2["coef"],
            "label_permutation_p": phase2["ri_leaveout"]["p_ri_two"],
            "pretrend_p": phase2["joint_pretrend_p"],
            "drop_cs": "wide interval includes zero",
            "note": "failed pretrend",
        },
        {
            "design": "P3 Task-fit x Trends",
            "gradient": "task fit by Google Trends",
            "outcome": "leave-out",
            "beta": p3["coef"],
            "label_permutation_p": phase3["ri_leaveout"]["p_ri_two"],
            "pretrend_p": phase3["joint_pretrend_p"],
            "drop_cs": "wide interval includes zero",
            "note": "failed pretrend",
        },
        {
            "design": "P4 Marker dose + category trends",
            "gradient": "category marker share",
            "outcome": "leave-out",
            "beta": trend_result["coef"],
            "label_permutation_p": trend_permutation_p,
            "pretrend_p": np.nan,
            "drop_cs": "not reported",
            "note": "category-specific trends",
        },
    ]
    pd.DataFrame(rows).to_csv(
        RESULTS / "q1_rq2_phase4_triangulation.csv", index=False
    )
    manifest = {
        "script": Path(__file__).name,
        "category_trend_result": trend_result,
        "label_permutation_p": trend_permutation_p,
        "permutation_draws": 2_000,
        "interpretation": (
            "Observational association only. The exposure designs do not pass "
            "all pretrend and reference-model diagnostics."
        ),
    }
    (RESULTS / "q1_rq2_phase4_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
