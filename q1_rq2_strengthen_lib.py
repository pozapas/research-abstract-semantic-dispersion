"""Shared utilities for the RQ2 strengthening effort (Phases 1-4).

Goal: evaluate the sensitivity of the field-month association with alternative
standard errors, a label-permutation reference model, leave-out outcomes, and
task-fit exposure specifications.

Nothing here re-embeds. Phase 1 rebuilds the eligible panel from the existing
result CSVs and exactly reproduces the published baseline
(beta=-0.002225, cluster_category p=0.0076) before any stress test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

FIRST_POST = pd.Period("2022-12", freq="M")
AI_HEAVY = {"cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"}


def build_eligible_panel(min_n: int = 15) -> pd.DataFrame:
    """Reproduce eligible_panel() from q1_build_expanded_rq2_event_study.py
    using only the published panel + exposure CSVs."""
    panel = pd.read_csv(RESULTS / "q1_rq2_expanded_month_panel.csv")
    exp = pd.read_csv(RESULTS / "q1_rq2_expanded_exposure_table.csv")
    expcols = [
        "primary_category", "macro_field", "nature_frequency_z",
        "nature_any_use_z", "nature_research_writing_z",
        "pre_task_fit_z", "pre_ai_ml_z",
    ]
    expcols = [c for c in expcols if c in exp.columns]
    frame = panel.merge(exp[expcols], on=["primary_category", "macro_field"], how="left")
    frame = frame[
        (frame.n_papers >= min_n)
        & frame.semantic_distance.notna()
        & frame.nature_frequency_z.notna()
    ].copy()
    mp = pd.PeriodIndex(frame.month, freq="M")
    frame["post_gpt"] = (mp >= FIRST_POST).astype(int)
    cnt = frame.groupby("primary_category")["post_gpt"].agg(
        pre=lambda s: int((s == 0).sum()), post=lambda s: int((s == 1).sum())
    )
    keep = cnt[(cnt.pre >= 24) & (cnt.post >= 18)].index
    frame = frame[frame.primary_category.isin(keep)].copy().reset_index(drop=True)
    return frame


def _design(d: pd.DataFrame, treat_col: str, control: str = "log_n_papers"):
    fe_f = pd.get_dummies(d.primary_category, prefix="f", drop_first=True, dtype=float)
    fe_t = pd.get_dummies(d.month.astype(str), prefix="t", drop_first=True, dtype=float)
    X = pd.concat(
        [d[[treat_col, control]].reset_index(drop=True),
         fe_f.reset_index(drop=True), fe_t.reset_index(drop=True)],
        axis=1,
    )
    X = X.loc[:, X.nunique() > 1]
    X.insert(0, "const", 1.0)
    return X.astype(float)


def fit_did(d: pd.DataFrame, exposure: str = "nature_frequency_z",
            outcome: str = "semantic_distance", cluster: str = "primary_category",
            weights: str = "n_papers"):
    """Two-way FE WLS DiD. Returns (fit, X, y, w, groups, treat_col)."""
    d = d.loc[d[outcome].notna()].copy().reset_index(drop=True)
    d["did_target"] = d[exposure].astype(float) * d["post_gpt"].astype(float)
    X = _design(d, "did_target")
    y = d[outcome].astype(float).reset_index(drop=True)
    w = (None if weights == "unweighted"
         else d[weights].astype(float).clip(lower=1.0).reset_index(drop=True))
    groups = d[cluster].astype(str).reset_index(drop=True)
    model = sm.OLS(y, X) if w is None else sm.WLS(y, X, weights=w)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return fit, X, y, w, groups, "did_target"


def did_summary(fit, term="did_target"):
    ci = fit.conf_int().loc[term]
    return dict(coef=float(fit.params[term]), se=float(fit.bse[term]),
               t=float(fit.tvalues[term]), p=float(fit.pvalues[term]),
               ci_low=float(ci.iloc[0]), ci_high=float(ci.iloc[1]))


def wild_cluster_bootstrap(d: pd.DataFrame, exposure="nature_frequency_z",
                           outcome="semantic_distance", cluster="macro_field",
                           weights="n_papers", B=999, seed=20260630):
    """Cameron-Gelbach-Miller wild cluster bootstrap, null imposed (H0: beta=0),
    Rademacher weights, cluster-robust t at the same `cluster` level.
    Appropriate when treatment is assigned at `cluster` granularity."""
    rng = np.random.default_rng(seed)
    d = d.loc[d[outcome].notna()].copy().reset_index(drop=True)
    d["did_target"] = d[exposure].astype(float) * d["post_gpt"].astype(float)
    X_full = _design(d, "did_target")
    y = d[outcome].astype(float).values
    w = (np.ones(len(d)) if weights == "unweighted"
         else d[weights].astype(float).clip(lower=1.0).values)
    gcol = d[cluster].astype(str).values
    clusters = np.unique(gcol)

    # observed t
    fit_full = sm.WLS(y, X_full, weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": gcol})
    t_obs = float(fit_full.tvalues["did_target"])

    # restricted fit (drop treatment) -> residuals + fitted under H0
    X_r = X_full.drop(columns=["did_target"])
    fit_r = sm.WLS(y, X_r, weights=w).fit()
    yhat_r = fit_r.fittedvalues.values
    resid_r = fit_r.resid.values

    t_star = np.empty(B)
    for b in range(B):
        signs = rng.choice([-1.0, 1.0], size=len(clusters))
        sign_map = dict(zip(clusters, signs))
        wts = np.array([sign_map[g] for g in gcol])
        y_star = yhat_r + wts * resid_r
        try:
            fb = sm.WLS(y_star, X_full, weights=w).fit(
                cov_type="cluster", cov_kwds={"groups": gcol})
            t_star[b] = fb.tvalues["did_target"]
        except Exception:
            t_star[b] = np.nan
    t_star = t_star[np.isfinite(t_star)]
    p = (np.sum(np.abs(t_star) >= abs(t_obs)) + 1) / (len(t_star) + 1)
    return dict(t_obs=t_obs, p_wild=float(p), n_boot=int(len(t_star)),
                n_clusters=int(len(clusters)))


def randomization_inference(d: pd.DataFrame, exposure="nature_frequency_z",
                            outcome="semantic_distance", assign_level="macro_field",
                            weights="n_papers", B=2000, seed=20260630, one_sided=True):
    """Permute the exposure label across assignment units (default macro_field),
    refit, and compare |beta| (or signed beta for one-sided). This respects that
    treatment is assigned at the field level, not the cell level."""
    rng = np.random.default_rng(seed)
    d = d.loc[d[outcome].notna()].copy().reset_index(drop=True)
    unit_z = d.groupby(assign_level)[exposure].first()
    units = unit_z.index.to_numpy()
    zvals = unit_z.to_numpy()

    fit_obs, *_ = fit_did(d, exposure=exposure, outcome=outcome,
                          cluster="primary_category", weights=weights)
    b_obs = float(fit_obs.params["did_target"])

    betas = np.empty(B)
    for b in range(B):
        permz = dict(zip(units, rng.permutation(zvals)))
        d2 = d.copy()
        d2[exposure] = d2[assign_level].map(permz).astype(float)
        try:
            fb, *_ = fit_did(d2, exposure=exposure, outcome=outcome,
                             cluster="primary_category", weights=weights)
            betas[b] = fb.params["did_target"]
        except Exception:
            betas[b] = np.nan
    betas = betas[np.isfinite(betas)]
    p_two = (np.sum(np.abs(betas) >= abs(b_obs)) + 1) / (len(betas) + 1)
    # one-sided in the observed (negative) direction
    if b_obs < 0:
        p_one = (np.sum(betas <= b_obs) + 1) / (len(betas) + 1)
    else:
        p_one = (np.sum(betas >= b_obs) + 1) / (len(betas) + 1)
    return dict(beta_obs=b_obs, p_ri_two=float(p_two), p_ri_one=float(p_one),
                n_perm=int(len(betas)), perm_mean=float(betas.mean()),
                perm_sd=float(betas.std()), n_units=int(len(units)))
