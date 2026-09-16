"""Phase 1: reference-model diagnostics for the field-month association.

Produces:
  results/q1_rq2_phase1_inference.csv      (all inference methods, one row each)
  results/q1_rq2_phase1_leaveout.csv       (leave-one-macro-out leverage table)
  results/q1_rq2_phase1_memo.md            (human-readable diagnostic memo)

The exposure was not randomized. Permutation p-values are diagnostics under
macro-field-label exchangeability.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from q1_rq2_strengthen_lib import (
    RESULTS, AI_HEAVY, build_eligible_panel, fit_did, did_summary,
    wild_cluster_bootstrap, randomization_inference,
)


def leave_one_macro_out(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_fit, *_ = fit_did(frame)
    base = did_summary(base_fit)
    rows.append(dict(dropped="(none) baseline", **base,
                     n_cat=frame.primary_category.nunique(),
                     n_macro=frame.macro_field.nunique(), n_cells=len(frame)))
    for macro in sorted(frame.macro_field.unique()):
        sub = frame[frame.macro_field != macro].copy()
        if sub.primary_category.nunique() < 10 or len(sub) < 100:
            rows.append(dict(dropped=f"drop {macro}", status="too_sparse"))
            continue
        try:
            fit, *_ = fit_did(sub)
            s = did_summary(fit)
            rows.append(dict(dropped=f"drop {macro}", **s,
                             n_cat=sub.primary_category.nunique(),
                             n_macro=sub.macro_field.nunique(), n_cells=len(sub)))
        except Exception as exc:
            rows.append(dict(dropped=f"drop {macro}", status=f"error:{exc}"))
    return pd.DataFrame(rows)


def main() -> None:
    frame = build_eligible_panel()

    # --- 0. reproduce baseline (validation gate) ---
    fit_cat, *_ = fit_did(frame, cluster="primary_category")
    base = did_summary(fit_cat)
    assert abs(base["coef"] + 0.002225) < 1e-5, f"baseline mismatch: {base}"
    print(f"[validated] baseline beta={base['coef']:.6f} p_cat={base['p']:.4f}")

    # --- 1. alternative inference on the SAME point estimate ---
    fit_macro, *_ = fit_did(frame, cluster="macro_field")
    macro = did_summary(fit_macro)

    wild = wild_cluster_bootstrap(frame, cluster="macro_field", B=1999)
    ri = randomization_inference(frame, assign_level="macro_field", B=4000)

    inf_rows = [
        dict(method="cluster_by_category (PUBLISHED)", coef=base["coef"], se=base["se"],
             p=base["p"], detail="102 clusters; treatment varies at macro level -> understates SE"),
        dict(method="cluster_by_macro_field", coef=macro["coef"], se=macro["se"],
             p=macro["p"], detail="12 clusters; too few for reliable CRVE asymptotics"),
        dict(method="wild_cluster_bootstrap (macro, null-imposed)", coef=base["coef"],
             se=np.nan, p=wild["p_wild"],
             detail=f"Rademacher, {wild['n_boot']} draws, {wild['n_clusters']} clusters"),
        dict(method="label_permutation (macro exposure, two-sided)",
             coef=ri["beta_obs"], se=np.nan, p=ri["p_ri_two"],
             detail=f"{ri['n_perm']} perms over {ri['n_units']} macro units"),
        dict(method="label_permutation (macro exposure, one-sided)",
             coef=ri["beta_obs"], se=np.nan, p=ri["p_ri_one"],
             detail=f"{ri['n_perm']} perms; directional hypothesis (contraction)"),
    ]
    inf = pd.DataFrame(inf_rows)
    inf.to_csv(RESULTS / "q1_rq2_phase1_inference.csv", index=False)

    # --- 2. leverage: leave-one-macro-out ---
    lo = leave_one_macro_out(frame)
    lo.to_csv(RESULTS / "q1_rq2_phase1_leaveout.csv", index=False)

    # drop-cs specific
    no_cs = frame[frame.macro_field != "cs"].copy()
    fit_nocs, *_ = fit_did(no_cs, cluster="primary_category")
    nocs = did_summary(fit_nocs)
    ri_nocs = randomization_inference(no_cs, assign_level="macro_field", B=4000)

    # --- decision ---
    decision = (
        "OBSERVATIONAL ONLY: report the point estimate with all reference-model "
        "diagnostics and do not claim randomized or causal identification."
    )

    memo = f"""# RQ2 Phase 1 — Reference-model diagnostics

## Validation
Baseline reproduced exactly: beta = {base['coef']:.6f}, cluster-by-category p = {base['p']:.4f}.

## The core problem
Treatment (Nature GenAI adoption) is assigned at the **macro-field** level with only
6 distinct values; cs = +3.55z carries 40/102 categories. Clustering by primary
category (102 clusters) treats correlated cells as independent and understates SEs.

## Inference on the same point estimate (beta = {base['coef']:.6f})
| Method | p-value |
|---|---|
| cluster-by-category (published) | {base['p']:.4f} |
| cluster-by-macro (12 clusters, unreliable) | {macro['p']:.4f} |
| **wild cluster bootstrap (macro)** | **{wild['p_wild']:.4f}** |
| **macro-label permutation (two-sided)** | **{ri['p_ri_two']:.4f}** |
| macro-label permutation (one-sided, contraction) | {ri['p_ri_one']:.4f} |

## Leverage (drop the cs macro entirely, 40 categories)
- beta = {nocs['coef']:.6f}, cluster-by-category p = {nocs['p']:.4f}
- macro-label permutation (two-sided) p = {ri_nocs['p_ri_two']:.4f}
- Direction {'PRESERVED' if nocs['coef'] < 0 else 'REVERSED'}; significance {'kept' if nocs['p'] < 0.05 else 'lost'} without cs.
- Full leave-one-macro-out table: results/q1_rq2_phase1_leaveout.csv

## Decision
{decision}
"""
    (RESULTS / "q1_rq2_phase1_memo.md").write_text(memo, encoding="utf-8")

    manifest = dict(
        phase="1_reference_model_diagnostics", date="2026-06-30",
        baseline=base, cluster_macro=macro, wild=wild, ri=ri,
        drop_cs=dict(summary=nocs, ri=ri_nocs), decision=decision,
    )
    (RESULTS / "q1_rq2_phase1_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8")

    print("\n" + memo)
    print("DECISION:", decision)


if __name__ == "__main__":
    main()
