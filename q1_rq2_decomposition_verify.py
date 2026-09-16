"""Verify the mean-pairwise-distance mixture decomposition (Theorem 1) on real
cells and produce the exact marker-composition split.

For a cell with n papers split into n_o marker-negative and n_g marker-positive
papers, the finite-sample mean pairwise distance obeys exactly:

  S(P) = w_oo S(A) + w_gg S(B) + w_og D(A,B),
  w_oo = n_o(n_o-1)/[n(n-1)], w_gg = n_g(n_g-1)/[n(n-1)], w_og = 2 n_o n_g/[n(n-1)],

where A is the marker-negative set, B is the marker-positive set, and D is the cross mean distance. The leave-out
statistic is S(A). We (1) confirm the identity holds to numerical error, and
(2) split the pre-to-post change into the leave-out change and the change in the
marker-composition gap.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIRST_POST = pd.Period("2022-12", freq="M")

GENAI = ["chatgpt","gpt-3","gpt-4","gpt4","openai","large language model","large language models",
         "llm","llms","generative ai","generative artificial intelligence","foundation model",
         "foundation models","prompt engineering","prompting"]


def pattern():
    parts = []
    for t in GENAI:
        e = re.escape(t.lower()).replace(r"\ ", r"\s+")
        parts.append(rf"(?<![A-Za-z0-9]){e}(?![A-Za-z0-9])" if t in {"llm","llms"} else rf"\b{e}\b")
    return re.compile("|".join(parts), re.I)


def main():
    usecols = ["id","primary_category","macro_field","title","abstract","submission_month","expanded_row_id"]
    df = pd.read_csv(DATA/"q1_rq2_expanded_stem_sample.csv", usecols=usecols, dtype={"id":str})
    df = df.sort_values("expanded_row_id").reset_index(drop=True)
    emb = np.load(DATA/"q1_rq2_expanded_embeddings.npy", mmap_mode="r")
    pat = pattern()
    txt = (df.title.fillna("")+" "+df.abstract.fillna("")).str.lower()
    df["marker"] = txt.map(lambda s: bool(pat.search(s)))
    mp = pd.PeriodIndex(df.submission_month, freq="M")
    df["post"] = (mp >= FIRST_POST).astype(int)

    def meandist(idx):
        if len(idx) < 2: return np.nan
        return float(np.mean(pdist(np.asarray(emb[idx], np.float32))))

    # (1) identity check on post cells with enough of both groups
    checks = []
    post = df[df.post == 1]
    for (cat, month), g in post.groupby(["primary_category","submission_month"]):
        io = g.loc[~g.marker, "expanded_row_id"].to_numpy(np.int64)
        ig = g.loc[g.marker, "expanded_row_id"].to_numpy(np.int64)
        no, ng, n = len(io), len(ig), len(g)
        if no < 15 or ng < 10:
            continue
        SA, SB = meandist(io), meandist(ig)
        D = float(np.mean(cdist(np.asarray(emb[io], np.float32), np.asarray(emb[ig], np.float32))))
        S_all = meandist(g.expanded_row_id.to_numpy(np.int64))
        w_oo = no*(no-1)/(n*(n-1)); w_gg = ng*(ng-1)/(n*(n-1)); w_og = 2*no*ng/(n*(n-1))
        recon = w_oo*SA + w_gg*SB + w_og*D
        checks.append(dict(cat=cat, month=str(month), n=n, no=no, ng=ng,
                           S_all=S_all, recon=recon, resid=S_all-recon,
                           SA=SA, SB=SB, D=D, pi=ng/n))
    ch = pd.DataFrame(checks)
    ch.to_csv(RESULTS/"q1_rq2_decomposition_identity_check.csv", index=False)
    print(f"identity check on {len(ch)} post cells")
    print(f"  max |residual| = {ch.resid.abs().max():.3e}  (should be ~0; exact identity)")
    print(f"  mean S(A) marker-negative = {ch.SA.mean():.4f} vs mean S(B) marker-positive = {ch.SB.mean():.4f}  "
          f"(marker-positive papers {'LESS' if ch.SB.mean()>ch.SA.mean() else 'MORE'} homogeneous / lower spread)")
    print(f"  mean marker share pi (these cells) = {ch.pi.mean():.3f}")

    # (2) Aggregate composition-gap vs leave-out change on the Phase-2 analysis
    #     population.  The same paper-count weights must be used for the full and
    #     leave-out statistics.  This makes
    #       Delta S = Delta A + Delta(S-A)
    #     an exact accounting identity for the reported weighted means.
    panel = pd.read_csv(RESULTS/"q1_rq2_phase2_panel.csv")
    gate = panel[(panel.n_non >= 15) & panel.dist_leaveout.notna()
                 & panel.dist_all.notna() & panel.cat_adoption.notna()].copy()
    counts = gate.groupby("primary_category")["post"].agg(
        pre=lambda s: int((s == 0).sum()),
        post=lambda s: int((s == 1).sum()),
    )
    keep = counts[(counts.pre >= 24) & (counts.post >= 18)].index
    el = gate[gate.primary_category.isin(keep)].copy()
    el["post"] = (pd.PeriodIndex(el.month, freq="M") >= FIRST_POST).astype(int)
    def wavg(s, sub): return np.average(sub[s], weights=sub["n_all"])
    pre, pos = el[el.post==0], el[el.post==1]
    S_pre = wavg("dist_all",pre); S_post = wavg("dist_all",pos)
    A_pre = wavg("dist_leaveout",pre); A_post = wavg("dist_leaveout",pos)
    dS = S_post - S_pre
    leaveout_change = A_post - A_pre
    composition_gap_change = (S_post - A_post) - (S_pre - A_pre)
    print("\nAggregate pre->post change (paper-weighted):")
    print(f"  dS(full)   = {dS:+.5f}")
    print(f"  leave-out change       = {leaveout_change:+.5f}  "
          f"({100*leaveout_change/dS:.1f}% of dS)")
    print(f"  composition-gap change = {composition_gap_change:+.5f}  "
          f"({100*composition_gap_change/dS:.1f}% of dS)")
    summary = dict(S_pre=S_pre,S_post=S_post,A_pre=A_pre,A_post=A_post,
                   dS_full=dS, leaveout_change=leaveout_change,
                   composition_gap_pre=S_pre-A_pre,
                   composition_gap_post=S_post-A_post,
                   composition_gap_change=composition_gap_change,
                   leaveout_fraction=leaveout_change/dS,
                   composition_gap_fraction=composition_gap_change/dS,
                   analysis_cells=int(len(el)),
                   analysis_categories=int(el.primary_category.nunique()),
                   analysis_papers=int(el.n_all.sum()),
                   identity_max_resid=float(ch.resid.abs().max()),
                   mean_SA=float(ch.SA.mean()), mean_SB=float(ch.SB.mean()))
    pd.Series(summary).to_csv(RESULTS/"q1_rq2_decomposition_summary.csv")
    print("\nsaved results/q1_rq2_decomposition_identity_check.csv + _summary.csv")


if __name__ == "__main__":
    main()
