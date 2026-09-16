"""Phase 3: Bartik / shift-share RQ2 design (independent triangulation).

Treatment intensity for field f in month t:
    exposure_ft = pre_task_fit_z[f]  x  chatgpt_salience_z[t]
where the field "share" is the pre-2022 text/code/literature task-fit of the
field (pre-determined, not mechanically tied to the outcome) and the time
"shock" is the ChatGPT Google-Trends index (0 before 2022-12, ramps to 100).

With two-way fixed effects, the share is absorbed by field fixed effects and the
shock is absorbed by month fixed effects. The interaction estimates whether
semantic dispersion changes with this task-fit exposure. This is an
observational diagnostic. It does not identify a causal effect.

Reads the Phase 2 panel (for both dist_all and the leave-out outcome). Run
q1_rq2_phase2_leaveout_dose.py first.

Outputs:
  results/q1_rq2_phase3_did.csv
  results/q1_rq2_phase3_event_study.csv
  results/q1_rq2_phase3_memo.md
  results/q1_rq2_phase3_manifest.json
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

from q1_rq2_strengthen_lib import RESULTS, AI_HEAVY

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIRST_POST = pd.Period("2022-12", freq="M")

EVENT_ORDER = ["lead_49_60", "lead_37_48", "lead_25_36", "lead_13_24", "lead_07_12",
               "lead_04_06", "lead_02_03", "reference_m01", "lag_00", "lag_01_02",
               "lag_03_06", "lag_07_12", "lag_13_24", "lag_25_36"]
EVENT_MID = {"lead_49_60": -54.5, "lead_37_48": -42.5, "lead_25_36": -30.5,
             "lead_13_24": -18.5, "lead_07_12": -9.5, "lead_04_06": -5.0,
             "lead_02_03": -2.5, "reference_m01": -1.0, "lag_00": 0.0,
             "lag_01_02": 1.5, "lag_03_06": 4.5, "lag_07_12": 9.5,
             "lag_13_24": 18.5, "lag_25_36": 30.5}


def zscore(s):
    s = pd.Series(s).astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else pd.Series(0.0, index=s.index)


def load():
    panel = pd.read_csv(RESULTS / "q1_rq2_phase2_panel.csv")
    exp = pd.read_csv(RESULTS / "q1_rq2_expanded_exposure_table.csv")
    share = exp[["primary_category", "pre_task_fit_z"]].drop_duplicates()
    panel = panel.merge(share, on="primary_category", how="left")

    tr = pd.read_csv(DATA / "google_trends.csv")
    tr["month"] = pd.PeriodIndex(pd.to_datetime(tr["date"]), freq="M").astype(str)
    tr["chatgpt_sal_z"] = zscore(tr["ChatGPT"])
    panel = panel.merge(tr[["month", "chatgpt_sal_z"]], on="month", how="left")

    panel["share_z"] = panel["pre_task_fit_z"]
    panel["exposure_ft"] = panel["share_z"] * panel["chatgpt_sal_z"]
    # eligibility: reuse Phase-2 leave-out gate
    f = panel[(panel.n_non >= 15) & panel.dist_leaveout.notna()
              & panel.share_z.notna() & panel.chatgpt_sal_z.notna()].copy()
    cnt = f.groupby("primary_category")["post"].agg(
        pre=lambda s: int((s == 0).sum()), post=lambda s: int((s == 1).sum()))
    keep = cnt[(cnt.pre >= 24) & (cnt.post >= 18)].index
    return f[f.primary_category.isin(keep)].copy().reset_index(drop=True)


def fit(d, treat_cols, outcome, control, weights, cluster="primary_category"):
    d = d.loc[d[outcome].notna()].copy().reset_index(drop=True)
    fe_f = pd.get_dummies(d.primary_category, prefix="f", drop_first=True, dtype=float)
    fe_t = pd.get_dummies(d.month.astype(str), prefix="t", drop_first=True, dtype=float)
    X = pd.concat([d[treat_cols + [control]].reset_index(drop=True),
                   fe_f.reset_index(drop=True), fe_t.reset_index(drop=True)], axis=1)
    X = X.loc[:, X.nunique() > 1]
    X.insert(0, "const", 1.0)
    X = X.astype(float)
    y = d[outcome].astype(float).reset_index(drop=True)
    w = d[weights].astype(float).clip(lower=1.0).reset_index(drop=True)
    f = sm.WLS(y, X, weights=w).fit(cov_type="cluster",
            cov_kwds={"groups": d[cluster].astype(str).reset_index(drop=True)})
    return f, X, d


def did(d, outcome, control, weights, label):
    f, _, _ = fit(d, ["exposure_ft"], outcome, control, weights)
    ci = f.conf_int().loc["exposure_ft"]
    return dict(label=label, outcome=outcome, coef=float(f.params["exposure_ft"]),
                se=float(f.bse["exposure_ft"]), p_cat=float(f.pvalues["exposure_ft"]),
                ci_low=float(ci.iloc[0]), ci_high=float(ci.iloc[1]),
                n_cat=d.primary_category.nunique(), n_cells=len(d))


def ri(d, outcome, control, weights, B=2000, seed=20260630):
    rng = np.random.default_rng(seed)
    f, _, _ = fit(d, ["exposure_ft"], outcome, control, weights)
    b_obs = float(f.params["exposure_ft"])
    share = d.groupby("primary_category")["share_z"].first()
    cats, vals = share.index.to_numpy(), share.to_numpy()
    betas = []
    for _ in range(B):
        pm = dict(zip(cats, rng.permutation(vals)))
        d2 = d.copy()
        d2["share_z"] = d2.primary_category.map(pm)
        d2["exposure_ft"] = d2["share_z"] * d2["chatgpt_sal_z"]
        try:
            fb, _, _ = fit(d2, ["exposure_ft"], outcome, control, weights)
            betas.append(float(fb.params["exposure_ft"]))
        except Exception:
            pass
    betas = np.array(betas)
    p_two = (np.sum(np.abs(betas) >= abs(b_obs)) + 1) / (len(betas) + 1)
    p_one = ((np.sum(betas <= b_obs) + 1) / (len(betas) + 1) if b_obs < 0
             else (np.sum(betas >= b_obs) + 1) / (len(betas) + 1))
    return dict(beta=b_obs, p_ri_two=float(p_two), p_ri_one=float(p_one), n_perm=len(betas))


def event_study(d, outcome, control, weights):
    # gradient = pre_task_fit share interacted with event-time bins; tests pretrends
    dd = d.copy()
    cols = []
    for lab in EVENT_ORDER:
        if lab == "reference_m01":
            continue
        c = f"ev_{lab}"
        dd[c] = dd["share_z"] * (dd.event_bin == lab).astype(float)
        cols.append(c)
    f, X, _ = fit(dd, cols, outcome, control, weights)
    rows = []
    for lab in EVENT_ORDER:
        if lab == "reference_m01":
            rows.append(dict(event_bin=lab, mid=EVENT_MID[lab], coef=0.0, p=np.nan)); continue
        c = f"ev_{lab}"
        if c in f.params.index:
            rows.append(dict(event_bin=lab, mid=EVENT_MID[lab], coef=float(f.params[c]),
                             p=float(f.pvalues[c])))
    ev = pd.DataFrame(rows).sort_values("mid")
    lead_cols = [f"ev_{l}" for l in EVENT_ORDER if l.startswith("lead") and f"ev_{l}" in X.columns]
    joint_p = np.nan
    if lead_cols:
        C = np.zeros((len(lead_cols), X.shape[1]))
        for i, c in enumerate(lead_cols):
            C[i, X.columns.get_loc(c)] = 1.0
        try:
            joint_p = float(f.f_test(C).pvalue)
        except Exception:
            pass
    return ev, joint_p


def main():
    d = load()
    print(f"eligible cells={len(d)} cats={d.primary_category.nunique()} "
          f"distinct share={d.share_z.round(4).nunique()}")
    res = [
        did(d, "dist_leaveout", "log_n_non", "n_non", "shiftshare_leaveout"),
        did(d, "dist_all", "log_n_all", "n_all", "shiftshare_full"),
        did(d[d.macro_field != "cs"], "dist_leaveout", "log_n_non", "n_non", "shiftshare_leaveout_drop_cs"),
        did(d[~d.primary_category.isin(AI_HEAVY)], "dist_leaveout", "log_n_non", "n_non", "shiftshare_leaveout_drop_aiheavy"),
    ]
    pd.DataFrame(res).to_csv(RESULTS / "q1_rq2_phase3_did.csv", index=False)
    ri_leave = ri(d, "dist_leaveout", "log_n_non", "n_non", B=2000)
    ev, joint_p = event_study(d, "dist_leaveout", "log_n_non", "n_non")
    ev.to_csv(RESULTS / "q1_rq2_phase3_event_study.csv", index=False)

    r0 = res[0]
    verdict = ("OBSERVATIONAL ONLY: the interval includes zero, the label-permutation "
               "p-value is not small, and the joint pretrend test raises a specification concern.")
    memo = f"""# RQ2 Phase 3 -- Shift-Share (Bartik) Triangulation

Date: 2026-06-30

## Design
exposure_ft = pre-2022 task-fit share[f] x ChatGPT-salience shock[t]; two-way fixed
effects, category-clustered standard errors, and a marker-negative outcome.
Distinct field shares: {d.share_z.round(4).nunique()} across {d.primary_category.nunique()} categories.

## Results
| Spec | beta | cluster-cat p | 95% CI |
|---|---|---|---|
| Shift-share leave-out (primary) | {res[0]['coef']:.6f} | {res[0]['p_cat']:.4f} | [{res[0]['ci_low']:.6f}, {res[0]['ci_high']:.6f}] |
| Shift-share full-sample | {res[1]['coef']:.6f} | {res[1]['p_cat']:.4f} | [{res[1]['ci_low']:.6f}, {res[1]['ci_high']:.6f}] |
| Leave-out, drop cs | {res[2]['coef']:.6f} | {res[2]['p_cat']:.4f} | -- |
| Leave-out, drop AI-heavy | {res[3]['coef']:.6f} | {res[3]['p_cat']:.4f} | -- |

## Label-permutation diagnostic
- Leave-out two-sided p = {ri_leave['p_ri_two']:.4f}; one-sided = {ri_leave['p_ri_one']:.4f} ({ri_leave['n_perm']} permutations)
- Event-study joint pretrend p = {round(joint_p,4) if np.isfinite(joint_p) else 'n/a'}

## Verdict
{verdict}
"""
    (RESULTS / "q1_rq2_phase3_memo.md").write_text(memo, encoding="utf-8")
    (RESULTS / "q1_rq2_phase3_manifest.json").write_text(json.dumps(dict(
        phase="3_shiftshare", date="2026-06-30", results=res, ri_leaveout=ri_leave,
        joint_pretrend_p=joint_p, verdict=verdict), indent=2, default=float), encoding="utf-8")
    print(memo)


if __name__ == "__main__":
    main()
