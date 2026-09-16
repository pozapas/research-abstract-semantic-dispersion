"""Compute fixed-composition descriptive pre/post contrasts for RQ1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main() -> None:
    panel = pd.read_csv(RESULTS / "q1_rq2_phase2_panel.csv")
    frame = panel[
        (panel.n_non >= 15)
        & panel.dist_leaveout.notna()
        & panel.dist_all.notna()
        & panel.cat_adoption.notna()
    ].copy()
    counts = frame.groupby("primary_category").post.agg(
        pre=lambda s: int((s == 0).sum()),
        post=lambda s: int((s == 1).sum()),
    )
    keep = counts[(counts.pre >= 24) & (counts.post >= 18)].index
    frame = frame[frame.primary_category.isin(keep)].copy()

    rows = []
    for outcome, weight, label in (
        ("dist_all", "n_all", "all marker statuses"),
        ("dist_leaveout", "n_non", "marker-negative only"),
    ):
        category_rows = []
        for category, group in frame.groupby("primary_category"):
            period_values = []
            for period in (0, 1):
                part = group[group.post == period]
                period_values.append(float(np.average(part[outcome], weights=part[weight])))
            category_rows.append(
                {
                    "primary_category": category,
                    "pre": period_values[0],
                    "post": period_values[1],
                    "pooled_weight": float(group[weight].sum()),
                }
            )
        by_category = pd.DataFrame(category_rows)
        fixed_weights = by_category.pooled_weight / by_category.pooled_weight.sum()
        pre = float(np.sum(fixed_weights * by_category.pre))
        post = float(np.sum(fixed_weights * by_category.post))
        rows.append(
            {
                "population": label,
                "pre_fixed_composition": pre,
                "post_fixed_composition": post,
                "delta": post - pre,
                "delta_pct": 100.0 * (post - pre) / pre,
                "categories": int(len(by_category)),
                "cells": int(len(frame)),
                "weight_definition": "each category receives its pooled-period paper share in both periods",
                "interpretation": "descriptive standardized contrast; no adjustment for the secular time trend",
            }
        )

    no_cs = frame[frame.macro_field != "cs"].copy()
    category_rows = []
    for category, group in no_cs.groupby("primary_category"):
        values = []
        for period in (0, 1):
            part = group[group.post == period]
            values.append(float(np.average(part.dist_leaveout, weights=part.n_non)))
        category_rows.append((category, values[0], values[1], float(group.n_non.sum())))
    by_category = pd.DataFrame(category_rows, columns=["category", "pre", "post", "weight"])
    fixed_weights = by_category.weight / by_category.weight.sum()
    pre = float(np.sum(fixed_weights * by_category.pre))
    post = float(np.sum(fixed_weights * by_category.post))
    rows.append(
        {
            "population": "marker-negative only; computer science excluded",
            "pre_fixed_composition": pre,
            "post_fixed_composition": post,
            "delta": post - pre,
            "delta_pct": 100.0 * (post - pre) / pre,
            "categories": int(len(by_category)),
            "cells": int(len(no_cs)),
            "weight_definition": "each category receives its pooled-period paper share in both periods",
            "interpretation": "descriptive standardized contrast; no adjustment for the secular time trend",
        }
    )

    output = pd.DataFrame(rows)
    output.to_csv(RESULTS / "q1_rq1_fixed_composition.csv", index=False)
    (RESULTS / "q1_rq1_fixed_composition_manifest.json").write_text(
        json.dumps(
            {
                "script": Path(__file__).name,
                "eligibility": "n_non >= 15; at least 24 pre months and 18 post months",
                "output": "results/q1_rq1_fixed_composition.csv",
                "claim_boundary": "descriptive only; the interrupted-time-series model supplies the trend adjustment",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
