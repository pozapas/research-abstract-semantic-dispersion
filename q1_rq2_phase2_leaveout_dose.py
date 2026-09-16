"""Phase 2: leave-out marker-dose RQ2 diagnostic.

This script addresses two limitations of the first specification:
  (1) It replaces the six-value macro-field exposure with each category's
      post-period marker share.
  (2) It recalculates semantic dispersion after it removes marker-positive
      papers. This reduces direct composition effects from these papers.

The model estimates an observational association. The label-permutation check
requires category-label exchangeability, and the event study tests whether the
pre-period patterns are compatible with the specification. Neither check gives
causal identification.

Outputs:
  results/q1_rq2_phase2_panel.csv
  results/q1_rq2_phase2_did.csv
  results/q1_rq2_phase2_event_study.csv
  results/q1_rq2_phase2_memo.md
  results/q1_rq2_phase2_manifest.json
"""
from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import pdist
from pathlib import Path

from q1_rq2_strengthen_lib import RESULTS, AI_HEAVY, _design

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIRST_POST = pd.Period("2022-12", freq="M")
MIN_N = 15

GENAI = ["chatgpt", "gpt-3", "gpt-4", "gpt4", "openai", "large language model",
         "large language models", "llm", "llms", "generative ai",
         "generative artificial intelligence", "foundation model",
         "foundation models", "prompt engineering", "prompting"]


def genai_pattern():
    parts = []
    for t in GENAI:
        e = re.escape(t.lower()).replace(r"\ ", r"\s+")
        parts.append(rf"(?<![A-Za-z0-9]){e}(?![A-Za-z0-9])" if t in {"llm", "llms"} else rf"\b{e}\b")
    return re.compile("|".join(parts), re.I)


def event_bin(t: int) -> str:
    edges = [(-49, "lead_49_60"), (-37, "lead_37_48"), (-25, "lead_25_36"),
             (-13, "lead_13_24"), (-7, "lead_07_12"), (-4, "lead_04_06"),
             (-2, "lead_02_03")]
    for thr, lab in edges:
        if t <= thr:
            return lab
    if t == -1: return "reference_m01"
    if t == 0: return "lag_00"
    if t <= 2: return "lag_01_02"
    if t <= 6: return "lag_03_06"
    if t <= 12: return "lag_07_12"
    if t <= 24: return "lag_13_24"
    return "lag_25_36"


EVENT_ORDER = ["lead_49_60", "lead_37_48", "lead_25_36", "lead_13_24", "lead_07_12",
               "lead_04_06", "lead_02_03", "reference_m01", "lag_00", "lag_01_02",
               "lag_03_06", "lag_07_12", "lag_13_24", "lag_25_36"]
EVENT_MID = {"lead_49_60": -54.5, "lead_37_48": -42.5, "lead_25_36": -30.5,
             "lead_13_24": -18.5, "lead_07_12": -9.5, "lead_04_06": -5.0,
             "lead_02_03": -2.5, "reference_m01": -1.0, "lag_00": 0.0,
             "lag_01_02": 1.5, "lag_03_06": 4.5, "lag_07_12": 9.5,
             "lag_13_24": 18.5, "lag_25_36": 30.5}


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else pd.Series(0.0, index=s.index)


def build_panel() -> pd.DataFrame:
    usecols = ["id", "primary_category", "macro_field", "title", "abstract",
               "submission_month", "expanded_row_id"]
    df = pd.read_csv(DATA / "q1_rq2_expanded_stem_sample.csv", usecols=usecols, dtype={"id": str})
    df = df.sort_values("expanded_row_id").reset_index(drop=True)
    if not np.array_equal(df.expanded_row_id.to_numpy(np.int64), np.arange(len(df))):
        raise ValueError("row ids not contiguous")
    emb = np.load(DATA / "q1_rq2_expanded_embeddings.npy", mmap_mode="r")
    if emb.shape[0] != len(df):
        raise ValueError("embeddings not aligned")

    pat = genai_pattern()
    txt = (df.title.fillna("") + " " + df.abstract.fillna("")).str.lower()
    df["marker"] = txt.map(lambda s: bool(pat.search(s)))
    mp = pd.PeriodIndex(df.submission_month, freq="M")
    df["month_period"] = mp
    df["post"] = (mp >= FIRST_POST).astype(int)
    df["event_time"] = mp.map(lambda p: p.ordinal - FIRST_POST.ordinal).astype(int)

    rows = []
    for (cat, month), grp in df.groupby(["primary_category", "month_period"], observed=True):
        idx_all = grp.expanded_row_id.to_numpy(np.int64)
        idx_non = grp.loc[~grp.marker, "expanded_row_id"].to_numpy(np.int64)
        n_all, n_non = len(idx_all), len(idx_non)

        def dist(idx):
            if len(idx) < 2:
                return np.nan
            v = np.asarray(emb[idx], dtype=np.float32)
            return float(np.mean(pdist(v, metric="euclidean")))

        et = int(month.ordinal - FIRST_POST.ordinal)
        rows.append(dict(
            primary_category=cat, macro_field=grp.macro_field.iloc[0],
            month=str(month), event_time=et, event_bin=event_bin(et),
            post=int(month >= FIRST_POST), n_all=n_all, n_non=n_non,
            n_marker=int(grp.marker.sum()), marker_share=float(grp.marker.mean()),
            dist_all=dist(idx_all), dist_leaveout=dist(idx_non),
            log_n_non=float(np.log1p(n_non)), log_n_all=float(np.log1p(n_all)),
        ))
    panel = pd.DataFrame(rows)

    # per-category adoption gradient = post-period mean marker share (pre-period ~0)
    adopt = (panel[panel.post == 1].groupby("primary_category")["marker_share"].mean()
             .rename("cat_adoption"))
    panel = panel.merge(adopt, on="primary_category", how="left")
    panel["cat_adoption_z"] = zscore(panel["cat_adoption"].fillna(0.0))
    panel.to_csv(RESULTS / "q1_rq2_phase2_panel.csv", index=False)
    return panel


def eligible(panel: pd.DataFrame) -> pd.DataFrame:
    f = panel[(panel.n_non >= MIN_N) & panel.dist_leaveout.notna()
              & panel.cat_adoption.notna()].copy()
    cnt = f.groupby("primary_category")["post"].agg(
        pre=lambda s: int((s == 0).sum()), post=lambda s: int((s == 1).sum()))
    keep = cnt[(cnt.pre >= 24) & (cnt.post >= 18)].index
    return f[f.primary_category.isin(keep)].copy().reset_index(drop=True)


def fit_generic(d, treat_col, outcome, control, cluster="primary_category", weights="n_non"):
    d = d.loc[d[outcome].notna()].copy().reset_index(drop=True)
    fe_f = pd.get_dummies(d.primary_category, prefix="f", drop_first=True, dtype=float)
    fe_t = pd.get_dummies(d.month.astype(str), prefix="t", drop_first=True, dtype=float)
    X = pd.concat([d[treat_col + [control]].reset_index(drop=True),
                   fe_f.reset_index(drop=True), fe_t.reset_index(drop=True)], axis=1)
    X = X.loc[:, X.nunique() > 1]
    X.insert(0, "const", 1.0)
    X = X.astype(float)
    y = d[outcome].astype(float).reset_index(drop=True)
    w = d[weights].astype(float).clip(lower=1.0).reset_index(drop=True)
    fit = sm.WLS(y, X, weights=w).fit(cov_type="cluster",
            cov_kwds={"groups": d[cluster].astype(str).reset_index(drop=True)})
    return fit, X, d


def ri_did(d, outcome, control, weights="n_non", B=2000, seed=20260630):
    """Permute category labels as a reference-model diagnostic."""
    rng = np.random.default_rng(seed)
    d = d.copy()
    d["did_t"] = d["cat_adoption_z"] * d["post"]
    fit, _, _ = fit_generic(d, ["did_t"], outcome, control, weights=weights)
    b_obs = float(fit.params["did_t"])
    cat_z = d.groupby("primary_category")["cat_adoption_z"].first()
    cats, zvals = cat_z.index.to_numpy(), cat_z.to_numpy()
    betas = []
    for _ in range(B):
        pm = dict(zip(cats, rng.permutation(zvals)))
        d2 = d.copy()
        d2["cat_adoption_z"] = d2.primary_category.map(pm)
        d2["did_t"] = d2["cat_adoption_z"] * d2["post"]
        try:
            fb, _, _ = fit_generic(d2, ["did_t"], outcome, control, weights=weights)
            betas.append(float(fb.params["did_t"]))
        except Exception:
            pass
    betas = np.array(betas)
    p_two = (np.sum(np.abs(betas) >= abs(b_obs)) + 1) / (len(betas) + 1)
    p_one = ((np.sum(betas <= b_obs) + 1) / (len(betas) + 1) if b_obs < 0
             else (np.sum(betas >= b_obs) + 1) / (len(betas) + 1))
    return dict(beta=b_obs, p_ri_two=float(p_two), p_ri_one=float(p_one), n_perm=len(betas))


def run_did(d, outcome, control, weights="n_non", label=""):
    dd = d.copy()
    dd["did_t"] = dd["cat_adoption_z"] * dd["post"]
    fit, _, _ = fit_generic(dd, ["did_t"], outcome, control, weights=weights)
    ci = fit.conf_int().loc["did_t"]
    return dict(label=label, outcome=outcome, coef=float(fit.params["did_t"]),
                se=float(fit.bse["did_t"]), p_cat=float(fit.pvalues["did_t"]),
                ci_low=float(ci.iloc[0]), ci_high=float(ci.iloc[1]),
                n_cat=dd.primary_category.nunique(), n_cells=len(dd))


def run_event_study(d, outcome, control, weights="n_non"):
    dd = d.copy()
    cols = []
    for lab in EVENT_ORDER:
        if lab == "reference_m01":
            continue
        c = f"ev_{lab}"
        dd[c] = dd["cat_adoption_z"] * (dd.event_bin == lab).astype(float)
        cols.append(c)
    fit, X, _ = fit_generic(dd, cols, outcome, control, weights=weights)
    rows = []
    for lab in EVENT_ORDER:
        if lab == "reference_m01":
            rows.append(dict(event_bin=lab, mid=EVENT_MID[lab], coef=0.0, p=np.nan,
                             ci_low=np.nan, ci_high=np.nan, reference=True))
            continue
        c = f"ev_{lab}"
        if c in fit.params.index:
            ci = fit.conf_int().loc[c]
            rows.append(dict(event_bin=lab, mid=EVENT_MID[lab], coef=float(fit.params[c]),
                             p=float(fit.pvalues[c]), ci_low=float(ci.iloc[0]),
                             ci_high=float(ci.iloc[1]), reference=False))
    ev = pd.DataFrame(rows).sort_values("mid")
    # joint pretrend test on lead terms
    lead_cols = [f"ev_{l}" for l in EVENT_ORDER if l.startswith("lead") and f"ev_{l}" in X.columns]
    if lead_cols:
        C = np.zeros((len(lead_cols), X.shape[1]))
        for i, c in enumerate(lead_cols):
            C[i, X.columns.get_loc(c)] = 1.0
        try:
            joint_p = float(fit.f_test(C).pvalue)
        except Exception:
            joint_p = np.nan
    else:
        joint_p = np.nan
    return ev, joint_p


def main():
    panel = build_panel()
    elig = eligible(panel)
    print(f"eligible cells={len(elig)} categories={elig.primary_category.nunique()} "
          f"distinct adoption={elig.cat_adoption_z.round(5).nunique()}")

    results = []
    # primary: leave-out outcome
    r_leave = run_did(elig, "dist_leaveout", "log_n_non", label="leaveout_primary")
    # negative control: full-sample outcome, same design
    r_full = run_did(elig, "dist_all", "log_n_all", weights="n_all", label="full_sample_control")
    # drop cs
    r_nocs = run_did(elig[elig.macro_field != "cs"], "dist_leaveout", "log_n_non", label="leaveout_drop_cs")
    # exclude AI-heavy
    r_noai = run_did(elig[~elig.primary_category.isin(AI_HEAVY)], "dist_leaveout", "log_n_non",
                     label="leaveout_drop_ai_heavy")
    # unweighted
    elig_uw = elig.copy(); elig_uw["unit"] = 1.0
    r_uw = run_did(elig_uw, "dist_leaveout", "log_n_non", weights="unit", label="leaveout_unweighted")
    results = [r_leave, r_full, r_nocs, r_noai, r_uw]
    did_df = pd.DataFrame(results)
    did_df.to_csv(RESULTS / "q1_rq2_phase2_did.csv", index=False)

    ri_leave = ri_did(elig, "dist_leaveout", "log_n_non", B=2000)
    ri_full = ri_did(elig, "dist_all", "log_n_all", weights="n_all", B=2000)

    ev, joint_p = run_event_study(elig, "dist_leaveout", "log_n_non")
    ev.to_csv(RESULTS / "q1_rq2_phase2_event_study.csv", index=False)

    verdict = ("OBSERVATIONAL ONLY: the label-permutation result is compatible with a negative "
               "association, but the joint pretrend test rejects parallel pre-period patterns; "
               "do not use a causal interpretation.")

    memo = f"""# RQ2 Phase 2 -- Leave-out Marker-Dose Design

Date: 2026-06-30

## Design
- Exposure: per-category post-period marker share (z), {elig.cat_adoption_z.round(5).nunique()} distinct values across {elig.primary_category.nunique()} categories.
- Outcome: mean pairwise SBERT distance among marker-negative papers.
- Two-way fixed effects (category and month), control log(1+n_non), weights n_non, and category-clustered standard errors.

## Main results
| Spec | beta | cluster-cat p | 95% CI |
|---|---|---|---|
| Leave-out (primary) | {r_leave['coef']:.6f} | {r_leave['p_cat']:.4f} | [{r_leave['ci_low']:.6f}, {r_leave['ci_high']:.6f}] |
| Full-sample (neg. control) | {r_full['coef']:.6f} | {r_full['p_cat']:.4f} | [{r_full['ci_low']:.6f}, {r_full['ci_high']:.6f}] |
| Leave-out, drop cs | {r_nocs['coef']:.6f} | {r_nocs['p_cat']:.4f} | [{r_nocs['ci_low']:.6f}, {r_nocs['ci_high']:.6f}] |
| Leave-out, drop AI-heavy | {r_noai['coef']:.6f} | {r_noai['p_cat']:.4f} | -- |
| Leave-out, unweighted | {r_uw['coef']:.6f} | {r_uw['p_cat']:.4f} | -- |

## Label-permutation diagnostic
- Leave-out two-sided p = {ri_leave['p_ri_two']:.4f}; one-sided = {ri_leave['p_ri_one']:.4f} ({ri_leave['n_perm']} permutations)
- Full-sample two-sided p = {ri_full['p_ri_two']:.4f}
- Event-study joint pretrend p = {joint_p if joint_p is None else round(joint_p,4)}

## Full-sample and leave-out comparison
- Full-sample beta = {r_full['coef']:.6f}; leave-out beta = {r_leave['coef']:.6f}
- Leave-out / full ratio = {(r_leave['coef']/r_full['coef']) if r_full['coef'] else float('nan'):.3f}
- This ratio is descriptive. It does not identify a mechanism or a spillover.

## Verdict
{verdict}
"""
    (RESULTS / "q1_rq2_phase2_memo.md").write_text(memo, encoding="utf-8")
    (RESULTS / "q1_rq2_phase2_manifest.json").write_text(json.dumps(dict(
        phase="2_leaveout_dose", date="2026-06-30",
        results=results, ri_leaveout=ri_leave, ri_full=ri_full,
        joint_pretrend_p=joint_p, verdict=verdict), indent=2, default=float), encoding="utf-8")
    print(memo)


if __name__ == "__main__":
    main()
