"""Generate the upgraded CreativeAI figure suite.

Fig. 1 is the manually maintained draw.io framework and is intentionally not
touched here. This script rebuilds the quantitative Nature-style figures:

  Fig. 2  Idea-space density atlas
  Fig. 3  Multi-lens contraction evidence
  Fig. 4  Interrupted-time-series temporal rupture
  Fig. 5  Adoption/exposure landscape
  Fig. 6  Exact marker-composition decomposition
  Fig. 7  Identification stress test
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde

import q1_fig_style as S
from q1_fig_style import LENSES, PAL, POST, PRE

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
S.set_theme()


def _summary(path: str) -> dict[str, float]:
    return pd.read_csv(R / path, index_col=0).iloc[:, 0].to_dict()


def _p_text(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n.a."
    p = float(p)
    if p < 1e-4:
        return r"$p<10^{-4}$"
    if p < 0.001:
        return "$p<0.001$"
    return f"$p={p:.3f}$"


def _sample(df: pd.DataFrame, n: int, seed: int = 11) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _shared_kde(pre: pd.DataFrame, post: pd.DataFrame, gridsize: int = 145):
    all_xy = pd.concat([pre[["pc1", "pc2"]], post[["pc1", "pc2"]]])
    xlo, xhi = all_xy.pc1.quantile([0.005, 0.995])
    ylo, yhi = all_xy.pc2.quantile([0.005, 0.995])
    xpad = 0.08 * (xhi - xlo)
    ypad = 0.08 * (yhi - ylo)
    xx, yy = np.mgrid[
        xlo - xpad:xhi + xpad:complex(gridsize),
        ylo - ypad:yhi + ypad:complex(gridsize),
    ]

    def kde(d: pd.DataFrame, seed: int):
        d = _sample(d, 5500, seed)
        xy = np.vstack([d.pc1.to_numpy(float), d.pc2.to_numpy(float)])
        z = gaussian_kde(xy)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        return z / z.max()

    return xx, yy, kde(pre, 21), kde(post, 22)


def _centroid_marker(ax, d: pd.DataFrame, color: str, label: str) -> tuple[float, float]:
    xy = d[["pc1", "pc2"]].to_numpy(float)
    center = xy.mean(axis=0)
    ax.scatter(center[0], center[1], s=150, color=color, ec="white", lw=1.8,
               zorder=7, label=label)
    ax.scatter(center[0], center[1], s=520, facecolor="none", edgecolor=color,
               lw=1.7, alpha=0.55, zorder=6)
    return float(center[0]), float(center[1])


def _short_category(label: str) -> str:
    return label.replace("astro-ph.", "ap.").replace("cond-mat.", "cm.").replace("physics.", "phys.")


def fig2_idea_space_atlas() -> None:
    d = pd.read_csv(R / "q1_idea_space_projection.csv", parse_dates=["analysis_date"])
    pre = d.loc[d.period.eq("Pre-GPT")]
    post = d.loc[d.period.eq("Post-GPT")]
    xx, yy, zpre, zpost = _shared_kde(pre, post)
    zdiff = zpost - zpre
    vmax = max(abs(float(zdiff.min())), abs(float(zdiff.max())))

    fig = plt.figure(figsize=(14.6, 9.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    panels = [
        (axes[0], zpre, PAL["blue"], "Pre-ChatGPT idea-space density", "a"),
        (axes[1], zpost, POST, "Post-ChatGPT idea-space density", "b"),
    ]
    for ax, z, color, ttl, letter in panels:
        ax.contourf(xx, yy, z, levels=np.linspace(0.02, 1, 12), cmap=S.cmap_from(color), zorder=1)
        ax.contour(xx, yy, z, levels=np.linspace(0.12, 0.95, 9), colors=[color],
                   linewidths=1.2, alpha=0.75, zorder=2)
        pts = _sample(pre if letter == "a" else post, 1400, 30 if letter == "a" else 31)
        ax.scatter(pts.pc1, pts.pc2, s=4, color=color, alpha=0.16, ec="none", zorder=3)
        S.title(ax, ttl, color)
        S.panel(ax, letter)
        ax.set_xlabel("Embedding projection 1")
        ax.set_ylabel("Embedding projection 2")
        S.polish(ax, grid_axis="")

    ax = axes[2]
    ax.contour(xx, yy, zpre, levels=np.linspace(0.18, 0.92, 7), colors=[PAL["blue"]],
               linewidths=1.4, alpha=0.72)
    ax.contour(xx, yy, zpost, levels=np.linspace(0.18, 0.92, 7), colors=[POST],
               linewidths=1.4, alpha=0.72)
    c0 = _centroid_marker(ax, pre, PAL["blue"], "Pre")
    c1 = _centroid_marker(ax, post, POST, "Post")
    ax.add_patch(FancyArrowPatch(c0, c1, arrowstyle="-|>", mutation_scale=20,
                                 lw=2.0, color=PAL["ink"], connectionstyle="arc3,rad=0.08",
                                 zorder=8))
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2,
              handletextpad=0.3, borderaxespad=0.0)
    S.title(ax, "Centroid drift over density contours", PAL["ink"])
    S.panel(ax, "c")
    ax.set_xlabel("Embedding projection 1")
    ax.set_ylabel("Embedding projection 2")
    S.polish(ax, grid_axis="")

    ax = axes[3]
    im = ax.contourf(xx, yy, zdiff, levels=np.linspace(-vmax, vmax, 17),
                     cmap=S.signed_cmap(), extend="both", zorder=1)
    ax.contour(xx, yy, zdiff, levels=[0], colors=[PAL["ink"]], linewidths=1.2, zorder=2)
    ax.scatter([], [], color=PAL["blue"], label="relative thinning")
    ax.scatter([], [], color=POST, label="relative concentration")
    ax.legend(loc="upper right")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.015)
    cb.set_label("Post density minus pre density", fontweight="bold")
    cb.formatter = FormatStrFormatter("%.2f")
    cb.update_ticks()
    S.title(ax, "Signed density reallocation", PAL["ink"])
    S.panel(ax, "d")
    ax.set_xlabel("Embedding projection 1")
    ax.set_ylabel("Embedding projection 2")
    S.polish(ax, grid_axis="")

    S.save(fig, "fig2_idea_space_atlas", h_pad=0.12, hspace=0.045)


def fig3_multilens_evidence() -> None:
    b = pd.read_csv(R / "q1_embedding_distance_bootstrap.csv")
    tdi = pd.read_csv(R / "q1_tdi_bootstrap.csv")
    sem_s = pd.read_csv(R / "q1_embedding_distance_summary.csv").iloc[0]
    tdi_s = pd.read_csv(R / "q1_tdi_summary.csv").iloc[0]
    ent = pd.read_csv(R / "q1_topic_stability.csv")
    rb = pd.read_csv(R / "q1_rq1_fixed_composition.csv")

    fig = plt.figure(figsize=(15.0, 10.0), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.25], height_ratios=[1, 1])
    ax_sem = fig.add_subplot(gs[0, 0])
    ax_tdi = fig.add_subplot(gs[0, 1])
    ax_ent = fig.add_subplot(gs[1, 0])
    ax_rb = fig.add_subplot(gs[1, 1])

    for xp, per, col, side in [(0, "Pre-GPT", PRE, "left"), (1, "Post-GPT", POST, "right")]:
        vals = b.loc[b.period.eq(per), "mean_distance"]
        S.half_violin(ax_sem, xp, vals, col, side=side, width=0.42)
        S.bees(ax_sem, xp, vals, col, seed=100 + xp)
    ax_sem.plot([0, 1], [
        b.loc[b.period.eq("Pre-GPT"), "mean_distance"].median(),
        b.loc[b.period.eq("Post-GPT"), "mean_distance"].median(),
    ], color=PAL["ink"], lw=2.0, ls=(0, (3, 2)))
    ax_sem.annotate(f"{sem_s.delta_pct:+.2f}%\n100 repeated samples",
                    xy=(0.5, max(b.mean_distance)), ha="center", va="top",
                    fontsize=12, color=POST, fontweight="bold")
    ax_sem.set_xticks([0, 1])
    ax_sem.set_xticklabels(["Pre", "Post"])
    ax_sem.set_ylabel("Mean pairwise SBERT distance")
    S.title(ax_sem, "Semantic-distance repeated samples", LENSES[0])
    S.panel(ax_sem, "a")
    S.polish(ax_sem)

    for xp, per, col, side in [(0, "Pre-GPT", PRE, "left"), (1, "Post-GPT", POST, "right")]:
        vals = tdi.loc[tdi.period.eq(per), "tdi_h1_persistence"]
        S.half_violin(ax_tdi, xp, vals, col, side=side, width=0.42)
        S.bees(ax_tdi, xp, vals, col, seed=110 + xp)
    ax_tdi.plot([0, 1], [
        tdi.loc[tdi.period.eq("Pre-GPT"), "tdi_h1_persistence"].median(),
        tdi.loc[tdi.period.eq("Post-GPT"), "tdi_h1_persistence"].median(),
    ], color=PAL["ink"], lw=2.0, ls=(0, (3, 2)))
    ax_tdi.annotate(f"{tdi_s.delta_pct:+.1f}%\n20 repeated samples",
                    xy=(0.5, max(tdi.tdi_h1_persistence)), ha="center", va="top",
                    fontsize=12, color=POST, fontweight="bold")
    ax_tdi.set_xticks([0, 1])
    ax_tdi.set_xticklabels(["Pre", "Post"])
    ax_tdi.set_ylabel(r"TDI $H_1$ persistence")
    S.title(ax_tdi, "Topological diversity loop persistence", LENSES[1])
    S.panel(ax_tdi, "b")
    S.polish(ax_tdi)

    labels = [
        f"mcs={int(row.min_cluster_size)}, ms={int(row.min_samples)}, "
        f"{str(row.noise_policy).replace('_', ' ')}"
        for _, row in ent.iterrows()
    ]
    y = np.arange(len(ent))[::-1]
    for yi, (_, row), label in zip(y, ent.iterrows(), labels):
        ax_ent.plot([row.pre_entropy, row.post_entropy], [yi, yi], color=PAL["grid"],
                    lw=12, solid_capstyle="round", zorder=1)
        ax_ent.plot([row.pre_entropy, row.post_entropy], [yi, yi], color=PAL["ink"],
                    lw=1.3, zorder=2)
        ax_ent.scatter(row.pre_entropy, yi, s=130, color=PRE, ec="white", lw=1.6, zorder=4)
        ax_ent.scatter(row.post_entropy, yi, s=130, color=POST, ec="white", lw=1.6, zorder=5)
        ax_ent.annotate(f"{row.entropy_delta_pct:+.2f}%",
                        xy=(max(row.pre_entropy, row.post_entropy), yi),
                        xytext=(10, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=12, color=POST, fontweight="bold")
    ax_ent.set_yticks(y)
    ax_ent.set_yticklabels(labels)
    ax_ent.set_xlabel("Shannon topic entropy")
    ax_ent.scatter([], [], s=110, color=PRE, label="Pre")
    ax_ent.scatter([], [], s=110, color=POST, label="Post")
    ent_leg = ax_ent.legend(loc="lower right", bbox_to_anchor=(0.98, 0.03), ncol=2,
                            borderaxespad=0.2, handletextpad=0.45,
                            columnspacing=0.9, frameon=True)
    ent_leg.get_frame().set_facecolor("white")
    ent_leg.get_frame().set_edgecolor(PAL["grid"])
    ent_leg.get_frame().set_alpha(0.94)
    ax_ent.set_ylim(-0.6, len(ent) - 0.4)
    S.title(ax_ent, "Topic entropy sensitivity; 68--74% noise", LENSES[2])
    S.panel(ax_ent, "c")
    S.polish(ax_ent, grid_axis="x")

    nice = ["All papers", "Marker-negative", "Marker-negative, excl. CS"]
    yy = np.arange(len(rb))[::-1]
    for yi, (_, row), lab in zip(yy, rb.iterrows(), nice):
        col = POST if row["delta_pct"] < 0 else PAL["muted"]
        ax_rb.plot([0, row["delta_pct"]], [yi, yi], color=col, lw=3.0,
                   solid_capstyle="round", alpha=0.78)
        ax_rb.scatter(row["delta_pct"], yi, s=145, color=col, ec="white", lw=1.5, zorder=5)
        ax_rb.annotate(f"{row['delta_pct']:+.2f}%", (row["delta_pct"], yi),
                       xytext=(7, 9), textcoords="offset points",
                       ha="left", va="bottom", fontsize=11,
                       color=col, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                 ec="none", alpha=0.94))
    ax_rb.axvline(0, color=PAL["ink"], lw=1.2)
    ax_rb.set_yticks(yy)
    ax_rb.set_yticklabels(nice)
    ax_rb.set_xlabel("Change in semantic spread (%)")
    S.title(ax_rb, "Fixed-composition descriptive contrasts", PAL["ink"])
    S.panel(ax_rb, "d")
    S.polish(ax_rb, grid_axis="x")

    S.save(fig, "fig3_multilens_evidence", h_pad=0.14, hspace=0.06)


def fig4_temporal_rupture() -> None:
    m = pd.read_csv(R / "q1_monthly_embedding_distance.csv", parse_dates=["month"])
    co = pd.read_csv(R / "q1_its_embedding_distance_coefficients.csv").set_index("term")["coef"].to_dict()

    def c(name: str) -> float:
        return float(co.get(name, 0.0))

    m["fit"] = c("const") + c("time") * m["time"] + c("post_gpt") * m["post_gpt"] + c("time_after_gpt") * m["time_after_gpt"]
    m["delta"] = m.distance.diff()
    cut = m.loc[m.post_gpt.eq(1), "month"].min()

    fig = plt.figure(figsize=(15.0, 8.9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.45, 1.0])
    ax_ts = fig.add_subplot(gs[0, :])
    ax_delta = fig.add_subplot(gs[1, 0])
    ax_slope = fig.add_subplot(gs[1, 1])
    ax_resid = fig.add_subplot(gs[1, 2])

    base = m.distance.min() - 0.004
    ax_ts.axvspan(cut, m.month.max(), color=POST, alpha=0.07, lw=0)
    ax_ts.fill_between(m.month, base, m.distance, color=PAL["sky"], alpha=0.12, zorder=1)
    ax_ts.plot(m.month, m.distance, color=PAL["slate"], lw=1.3, alpha=0.72, zorder=3)
    sizes = np.clip(m.n_papers / 8, 26, 165)
    ax_ts.scatter(m.month, m.distance, s=sizes, color=PAL["blue"], alpha=0.54,
                  ec="white", lw=0.8, zorder=4)
    for mask, col in [(m.post_gpt.eq(0), PAL["blue"]), (m.post_gpt.eq(1), POST)]:
        d = m.loc[mask]
        ax_ts.plot(d.month, d.fit, color=col, lw=3.2, zorder=5)
    ax_ts.axvline(cut, color=PAL["ink"], lw=1.4, ls=(0, (4, 3)))
    ax_ts.annotate("ChatGPT release\n30 Nov 2022", xy=(cut, ax_ts.get_ylim()[1]),
                   xytext=(8, -8), textcoords="offset points",
                   ha="left", va="top", fontsize=12, fontweight="bold")
    slope_target_date = pd.Timestamp("2024-08-01")
    slope_target = m.loc[(m.month - slope_target_date).abs().idxmin()]
    ax_ts.annotate(f"slope change {c('time_after_gpt'):+.2e}\nHAC {_p_text(1e-5)}",
                   xy=(slope_target.month, slope_target.fit),
                   xytext=(pd.Timestamp("2024-10-01"), slope_target.fit + 0.017),
                   textcoords="data", ha="left", va="bottom",
                   fontsize=12, color=POST, fontweight="bold",
                   arrowprops=dict(arrowstyle="-", color=POST, lw=1.4,
                                   shrinkA=2, shrinkB=2))
    ax_ts.set_ylabel("Mean pairwise SBERT distance")
    ax_ts.set_xlabel("Month")
    ax_ts.xaxis.set_major_locator(mdates.YearLocator())
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    S.title(ax_ts, "Interrupted time series, 2018-2025", PAL["ink"])
    S.panel(ax_ts, "a")
    S.polish(ax_ts)

    pre_d = m.loc[m.post_gpt.eq(0), "delta"].dropna()
    post_d = m.loc[m.post_gpt.eq(1), "delta"].dropna()
    S.half_violin(ax_delta, 0, pre_d, PAL["blue"], side="left", width=0.44)
    S.half_violin(ax_delta, 1, post_d, POST, side="right", width=0.44)
    S.bees(ax_delta, 0, pre_d, PAL["blue"], seed=40)
    S.bees(ax_delta, 1, post_d, POST, seed=41)
    ax_delta.axhline(0, color=PAL["ink"], lw=1.1)
    ax_delta.set_xticks([0, 1])
    ax_delta.set_xticklabels(["Pre", "Post"])
    ax_delta.set_ylabel("Month-to-month change")
    S.title(ax_delta, "Local monthly increments", PAL["ink"])
    S.panel(ax_delta, "b")
    S.polish(ax_delta)

    slopes = pd.DataFrame({
        "phase": ["Pre trend", "Post trend"],
        "slope": [c("time"), c("time") + c("time_after_gpt")],
        "color": [PAL["blue"], POST],
    })
    ax_slope.axhline(0, color=PAL["ink"], lw=1.1)
    for i, row in slopes.iterrows():
        ax_slope.plot([i, i], [0, row.slope], color=row.color, lw=8,
                      solid_capstyle="round")
        ax_slope.scatter(i, row.slope, s=170, color=row.color, ec="white", lw=1.5, zorder=5)
        ax_slope.annotate(f"{row.slope:+.2e}", (i, row.slope),
                          xytext=(18, 0),
                          textcoords="offset points", ha="left",
                          va="center",
                          fontsize=12, color=row.color, fontweight="bold")
    ax_slope.set_xticks([0, 1])
    ax_slope.set_xticklabels(slopes.phase)
    ax_slope.set_xlim(-0.42, 1.42)
    ax_slope.set_ylim(min(slopes.slope) * 1.13, 0.00005)
    ax_slope.set_ylabel("Monthly fitted slope")
    S.title(ax_slope, "Slope anatomy", PAL["ink"])
    S.panel(ax_slope, "c")
    S.polish(ax_slope)

    m["resid"] = m.distance - m.fit
    ax_resid.scatter(m.fit, m.resid, s=sizes, c=np.where(m.post_gpt.eq(1), POST, PAL["blue"]),
                     alpha=0.62, ec="white", lw=0.7)
    ax_resid.axhline(0, color=PAL["ink"], lw=1.1)
    ax_resid.set_xlabel("Fitted semantic spread")
    ax_resid.set_ylabel("Residual")
    S.title(ax_resid, "Fit diagnostics", PAL["ink"])
    S.panel(ax_resid, "d")
    S.polish(ax_resid)

    S.save(fig, "fig4_temporal_rupture", h_pad=0.13, hspace=0.055)


def fig5_exposure_landscape() -> None:
    exp = pd.read_csv(R / "q1_rq2_expanded_exposure_table.csv")
    panel = pd.read_csv(R / "q1_rq2_expanded_month_panel.csv")
    nat = pd.read_csv(R / "q1_nature_genai_adoption_by_field.csv")

    fig = plt.figure(figsize=(17.2, 12.8), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.38, 1.02], height_ratios=[1.48, 1.58])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_macro = fig.add_subplot(gs[1, 1])

    marker_pct = exp.observed_genai_marker_share * 100
    sizes = 28 + 9.5 * np.sqrt(exp.rows.clip(lower=1))
    sc = ax_map.scatter(exp.nature_frequency_z, exp.pre_task_fit_z, s=sizes,
                        c=marker_pct, cmap=S.signed_cmap(), vmin=0,
                        vmax=max(2.5, marker_pct.quantile(0.98)),
                        alpha=0.76, ec="white", lw=1.0)
    ax_map.axvline(0, color=PAL["ink"], lw=1.1, ls=(0, (3, 3)))
    ax_map.axhline(0, color=PAL["ink"], lw=1.1, ls=(0, (3, 3)))
    top = exp.assign(marker_pct=marker_pct).sort_values("marker_pct", ascending=False).head(9)
    for _, row in top.iterrows():
        ax_map.annotate(_short_category(row.primary_category),
                        (row.nature_frequency_z, row.pre_task_fit_z),
                        xytext=(7, 5), textcoords="offset points",
                        fontsize=10.5, color=PAL["ink"], fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=PAL["grid"], lw=0.8))
    xpad_left = 0.28
    xpad_right = 0.78
    ypad = 0.28
    ax_map.set_xlim(exp.nature_frequency_z.min() - xpad_left,
                    exp.nature_frequency_z.max() + xpad_right)
    ax_map.set_ylim(exp.pre_task_fit_z.min() - ypad,
                    exp.pre_task_fit_z.max() + ypad)
    cb = fig.colorbar(sc, ax=ax_map, shrink=0.70, pad=0.085, fraction=0.035)
    cb.set_label("Observed GenAI-marker share (%)", fontweight="bold")
    ax_map.set_xlabel("External Nature-survey adoption exposure (z)")
    ax_map.set_ylabel("Pre-ChatGPT task-fit exposure (z)")
    S.title(ax_map, "Field-level exposure cartography", PAL["ink"])
    S.note(ax_map, "Bubble area is proportional to category sample size",
           xy=(0.97, 0.035), ha="right", va="bottom")
    S.panel(ax_map, "a")
    S.polish(ax_map)

    use_cols = [
        "share_use_research", "share_use_code", "share_use_manuscripts",
        "share_use_review", "share_use_literature", "share_use_scientific_search",
        "share_use_brainstorm",
    ]
    use_labs = ["Research", "Code", "Manuscript", "Review", "Literature", "Search", "Brainstorm"]
    nat = nat.sort_values("mean_frequency_score", ascending=False)
    mat = nat[use_cols].to_numpy(float) * 100
    heat_cmap = LinearSegmentedColormap.from_list(
        "nature_use_heat",
        ["#F7FAFC", "#DDEDEA", "#9DD2CA", "#42AFA4", "#08766F"],
        N=256,
    )
    im = ax_heat.imshow(mat, aspect="auto", cmap=heat_cmap, vmin=5, vmax=42)
    ax_heat.set_xticks(np.arange(-.5, len(use_labs), 1), minor=True)
    ax_heat.set_yticks(np.arange(-.5, len(nat), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    ax_heat.set_yticks(np.arange(len(nat)))
    ax_heat.set_yticklabels(nat.nature_field)
    ax_heat.set_xticks(np.arange(len(use_labs)))
    ax_heat.set_xticklabels(use_labs, rotation=32, ha="right")
    ax_heat.tick_params(axis="x", pad=2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt_col = "white" if mat[i, j] >= 31 else PAL["ink"]
            ax_heat.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                         fontsize=8.4, color=txt_col, fontweight="bold",
                         path_effects=[pe.withStroke(linewidth=2.1, foreground="white", alpha=0.55)]
                         if txt_col != "white" else
                         [pe.withStroke(linewidth=1.6, foreground=PAL["ink"], alpha=0.45)])
    cb2 = fig.colorbar(im, ax=ax_heat, shrink=0.86, pad=0.018, fraction=0.040)
    cb2.set_label("Use share (%)", fontweight="bold")
    S.title(ax_heat, "Nature survey use-channel fingerprint", PAL["teal"])
    S.panel(ax_heat, "b")
    ax_heat.grid(False)

    macro = panel.groupby(["macro_field", "post_gpt"], as_index=False).semantic_distance.mean()
    wide = macro.pivot(index="macro_field", columns="post_gpt", values="semantic_distance").dropna()
    wide.columns = ["pre", "post"]
    wide["pct"] = (wide["post"] - wide["pre"]) / wide["pre"] * 100
    wide = wide.sort_values("pct")
    y = np.arange(len(wide))[::-1]
    for yi, (field, row) in zip(y, wide.iterrows()):
        col = POST if row.pct < 0 else PAL["teal"]
        ax_macro.plot([row.pre, row.post], [yi, yi], color=PAL["grid"], lw=10,
                      solid_capstyle="round", zorder=1)
        ax_macro.plot([row.pre, row.post], [yi, yi], color=col, lw=2.0, zorder=2)
        ax_macro.scatter(row.pre, yi, s=90, color=PRE, ec="white", lw=1.1, zorder=3)
        ax_macro.scatter(row.post, yi, s=90, color=col, ec="white", lw=1.1, zorder=4)
        ax_macro.annotate(f"{row.pct:+.1f}%", (max(row.pre, row.post), yi),
                          xytext=(8, 0), textcoords="offset points",
                          va="center", fontsize=10.5, color=col, fontweight="bold")
    ax_macro.set_yticks(y)
    ax_macro.set_yticklabels(wide.index)
    ax_macro.set_xlabel("Mean semantic spread")
    S.title(ax_macro, "Macro-field pre/post shift", PAL["ink"])
    S.panel(ax_macro, "c")
    S.polish(ax_macro, grid_axis="x")

    S.save(fig, "fig5_exposure_landscape", h_pad=0.055, hspace=0.025)


def _ribbon(ax, start, end, width, color, rad=0.0, alpha=0.72) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-", mutation_scale=1,
                                 connectionstyle=f"arc3,rad={rad}", lw=width,
                                 color=color, alpha=alpha, capstyle="round"))


def fig6_decomposition_engine() -> None:
    summ = _summary("q1_rq2_decomposition_summary.csv")
    ck = pd.read_csv(R / "q1_rq2_decomposition_identity_check.csv")

    fig = plt.figure(figsize=(15.2, 5.9), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.14])
    ax_exact = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])

    sc = ax_exact.scatter(ck.S_all, ck.recon, s=48, c=ck.pi,
                          cmap=S.cmap_from(PAL["blue"], 0.95),
                          alpha=0.78, ec="white", lw=0.55)
    lo, hi = min(ck.S_all.min(), ck.recon.min()), max(ck.S_all.max(), ck.recon.max())
    ax_exact.plot([lo, hi], [lo, hi], color=PAL["ink"], lw=1.4, ls=(0, (4, 3)))
    ax_exact.set_xlabel("Observed aggregate spread S(P)")
    ax_exact.set_ylabel("Reconstructed from Theorem 1")
    ax_exact.text(0.04, 0.96, r"$S(P)=w_{oo}S(A)+w_{gg}S(B)+w_{og}D(A,B)$",
                  transform=ax_exact.transAxes, ha="left", va="top",
                  fontsize=12.5, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.34", fc="white",
                            ec=PAL["grid"], lw=1.0, alpha=0.94))
    ax_exact.text(0.05, 0.84,
                  f"{len(ck):,} cells; max |residual| = {ck.resid.abs().max():.1e}",
                  transform=ax_exact.transAxes, ha="left", va="top",
                  fontsize=11.5, color=PAL["slate"])
    cb = fig.colorbar(sc, ax=ax_exact, shrink=0.78, pad=0.018, fraction=0.046)
    cb.set_label("Marker-positive paper share in cell", fontweight="bold")
    S.title(ax_exact, "Empirical cell identity check", PAL["ink"])
    S.panel(ax_exact, "a")
    S.polish(ax_exact)

    for xp, vals, col, side, lab in [
        (0, ck.SA, PAL["green"], "left", "Marker-negative\nS(A)"),
        (1, ck.SB, POST, "right", "Marker-positive\nS(B)"),
        (2, ck.D, PAL["gold"], "right", "Between\nD(A,B)"),
    ]:
        S.half_violin(ax_dist, xp, vals, col, side=side, width=0.36)
        S.bees(ax_dist, xp, vals, col, seed=200 + xp, size=12, alpha=0.22)
    ax_dist.set_xticks([0, 1, 2])
    ax_dist.set_xticklabels(["Marker-negative\nS(A)", "Marker-positive\nS(B)", "Between\nD(A,B)"])
    ax_dist.set_ylabel("Cell-level spread component")
    diff_pct = (summ["mean_SB"] / summ["mean_SA"] - 1) * 100
    ax_dist.annotate(f"Marker-positive cells lower\n{diff_pct:.1f}% vs marker-negative",
                     xy=(1, ck.SB.mean()), xytext=(1.30, ck.SB.mean() - 0.055),
                     textcoords="data", ha="left", va="center",
                     fontsize=12, color=POST, fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=POST, lw=1.2))
    S.title(ax_dist, "Marker-group cell distributions", PAL["ink"])
    S.panel(ax_dist, "b")
    S.polish(ax_dist)

    S.save(fig, "fig6_decomposition_engine", wspace=0.09, h_pad=0.18)


def fig7_identification_stress_test() -> None:
    tri = pd.read_csv(R / "q1_rq2_phase4_triangulation.csv")
    ev = pd.read_csv(R / "q1_rq2_phase2_event_study.csv")
    inf = pd.read_csv(R / "q1_rq2_phase1_inference.csv")

    fig = plt.figure(figsize=(15.5, 10.8), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.18])
    ax_const = fig.add_subplot(gs[0, 0])
    ax_event = fig.add_subplot(gs[0, 1])
    gs_inf = gs[1, 0].subgridspec(2, 1, height_ratios=[1.0, 0.105], hspace=0.045)
    gs_led = gs[1, 1].subgridspec(2, 1, height_ratios=[1.0, 0.105], hspace=0.045)
    ax_inf = fig.add_subplot(gs_inf[0, 0])
    ax_inf_leg = fig.add_subplot(gs_inf[1, 0])
    ax_led = fig.add_subplot(gs_led[0, 0])
    ax_led_leg = fig.add_subplot(gs_led[1, 0])

    threshold = -math.log10(0.05)
    tri["x"] = -np.log10(tri.label_permutation_p.astype(float))
    tri["pretrend_num"] = pd.to_numeric(tri.pretrend_p, errors="coerce")
    tri["y"] = -np.log10(tri.pretrend_num)
    tri.loc[tri.y.isna(), "y"] = 0.08
    ax_const.axvspan(threshold, tri.x.max() + 0.45, ymin=0, ymax=threshold / (tri.y.max() + 0.45),
                     color=PAL["green"], alpha=0.09, lw=0)
    ax_const.axvline(threshold, color=PAL["ink"], lw=1.2, ls=(0, (4, 3)))
    ax_const.axhline(threshold, color=PAL["ink"], lw=1.2, ls=(0, (4, 3)))
    for _, row in tri.iterrows():
        col = POST if (
            "failed" in str(row.note).lower()
            or row.pretrend_num < 0.05
            or row.label_permutation_p >= 0.05
        ) else PAL["green"]
        ax_const.scatter(row.x, row.y, s=210, color=col, ec="white", lw=1.6, zorder=4)
        ax_const.annotate(row.design.split(" ")[0], (row.x, row.y), xytext=(8, 6),
                          textcoords="offset points", fontsize=12, fontweight="bold", color=col)
    ax_const.set_xlabel(r"Label-permutation diagnostic: $-\log_{10}(p)$")
    ax_const.set_ylabel(r"Pretrend stress: $-\log_{10}$(pretrend p)")
    ax_const.set_xlim(0.05, tri.x.max() + 0.55)
    ax_const.set_ylim(-0.06, tri.y.max() + 0.30)
    ax_const.text(0.04, 0.92,
                  "Preferred diagnostic region:\nlower-right; no design lands there",
                  transform=ax_const.transAxes, ha="left", va="top",
                  fontsize=11.2, color=PAL["slate"])
    S.title(ax_const, "Two-screen design validity map", PAL["ink"])
    S.panel(ax_const, "a")
    S.polish(ax_const)

    pre = ev.loc[ev.mid < 0]
    post = ev.loc[ev.mid >= 0]
    ax_event.axvspan(ev.mid.min() - 4, -0.5, color=PAL["blue"], alpha=0.07, lw=0)
    for d, col in [(pre, PAL["blue"]), (post, POST)]:
        yerr = np.vstack([d.coef - d.ci_low, d.ci_high - d.coef])
        ax_event.errorbar(d.mid, d.coef, yerr=yerr, fmt="o", ms=7,
                          color=col, ecolor=col, elinewidth=1.5, capsize=0, lw=0)
    ax_event.axhline(0, color=PAL["ink"], lw=1.2)
    ax_event.axvline(-0.5, color=PAL["ink"], lw=1.3, ls=(0, (4, 3)))
    ax_event.annotate("pretrend failure\n$p=0.005$", xy=(pre.mid.median(), ax_event.get_ylim()[1]),
                      xytext=(0, -10), textcoords="offset points", ha="center", va="top",
                      fontsize=12, color=PAL["blue"], fontweight="bold")
    ax_event.set_xlabel("Event time from cutoff (months)")
    ax_event.set_ylabel("Adoption x period coefficient")
    S.title(ax_event, "Exposure-gradient event-study diagnostics", PAL["ink"])
    S.panel(ax_event, "b")
    S.polish(ax_event)

    inf = inf.loc[inf.p.notna()].copy()
    inf["pnum"] = inf.p.astype(float)
    inf["method_short"] = [
        "Category CRVE", "Macro CRVE", "Wild cluster", "Macro-label perm.", "One-sided perm."
    ][:len(inf)]
    inf["score_raw"] = -np.log10(inf.pnum)
    cap = 5.0
    inf["score"] = inf.score_raw.clip(upper=cap)
    inf = inf.sort_values("score_raw", ascending=False)
    yy = np.arange(len(inf))[::-1]
    ax_inf.axvline(threshold, color=PAL["ink"], lw=1.2, ls=(0, (4, 3)))
    for yi, (_, row) in zip(yy, inf.iterrows()):
        col = POST if row.pnum < 0.05 else PAL["blue"]
        ax_inf.plot([0, row.score], [yi, yi], color=col, lw=3,
                    solid_capstyle="round")
        marker = ">" if row.score_raw > cap else "o"
        ax_inf.scatter(row.score, yi, s=150, marker=marker, color=col,
                       ec="white", lw=1.4, zorder=4)
        ax_inf.annotate(_p_text(row.pnum), (row.score, yi), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=11,
                        color=col, fontweight="bold")
    ax_inf.set_yticks(yy)
    ax_inf.set_yticklabels(inf.method_short)
    ax_inf.set_xlim(0, cap + 0.85)
    ax_inf.set_xlabel(r"Inference evidence: $-\log_{10}(p)$ (capped at 5)")
    S.title(ax_inf, "Inference depends on the reference model", PAL["ink"])
    S.panel(ax_inf, "c")
    S.polish(ax_inf, grid_axis="x")
    ax_inf_leg.axis("off")
    inf_handles = [
        Line2D([0], [0], color=POST, lw=2.8, marker="o", markerfacecolor=POST,
               markeredgecolor="white", markeredgewidth=1.0, label=r"$p<0.05$"),
        Line2D([0], [0], color=PAL["blue"], lw=2.8, marker="o",
               markerfacecolor=PAL["blue"], markeredgecolor="white",
               markeredgewidth=1.0, label=r"$p\geq0.05$"),
        Line2D([0], [0], color=PAL["ink"], lw=1.2, ls=(0, (4, 3)),
               label="0.05 threshold"),
    ]
    inf_leg = ax_inf_leg.legend(handles=inf_handles, loc="center", ncol=3,
                                fontsize=9.2, handlelength=1.8,
                                columnspacing=1.1, handletextpad=0.5,
                                borderaxespad=0.0, frameon=True)
    inf_leg.get_frame().set_facecolor("white")
    inf_leg.get_frame().set_edgecolor(PAL["grid"])
    inf_leg.get_frame().set_alpha(0.94)

    gate_x = np.arange(3, dtype=float)
    gate_labels = [
        "Permutation\n$p<0.05$",
        "Pretrend\n$p>0.05$",
        "No-CS CI\nexcludes zero",
    ]
    design_names = {
        "P1": "Nature adoption",
        "P2": "Marker dose",
        "P3": "Shift-share",
        "P4": "Marker+trends",
    }
    design_cols = {
        "P1": PAL["blue"],
        "P2": POST,
        "P3": PAL["gold"],
        "P4": PAL["purple"],
    }
    x_jitter = {"P1": -0.045, "P2": -0.015, "P3": 0.015, "P4": 0.045}
    ax_led.axhspan(0, 1.05, color=PAL["green"], alpha=0.075, lw=0)
    ax_led.axhspan(-1.22, 0, color=POST, alpha=0.060, lw=0)
    ax_led.axhline(0, color=PAL["ink"], lw=1.25, ls=(0, (4, 3)))
    for gx in gate_x:
        ax_led.axvline(gx, color=PAL["grid"], lw=1.0, zorder=0)
    ax_led.text(-0.27, 0.90, "pass side", color=PAL["green"],
                fontsize=10.5, fontweight="bold")
    ax_led.text(-0.27, -1.08, "fail side", color=POST,
                fontsize=10.5, fontweight="bold")

    for _, row in tri.iterrows():
        permutation_p = float(row.label_permutation_p)
        pretrend_p = row.pretrend_num
        drop_ok = not ("loses" in str(row.drop_cs).lower())
        code = row.design.split(" ")[0]
        vals = np.array([
            np.log10(0.05 / permutation_p),
            np.nan if pd.isna(pretrend_p) else np.log10(float(pretrend_p) / 0.05),
            0.62 if drop_ok else -0.62,
        ], dtype=float)
        xvals = gate_x + x_jitter[code]
        finite = np.isfinite(vals)
        shown_vals = np.where(finite, vals, 0.0)
        line_col = design_cols[code]
        line_style = (0, (2, 2)) if not finite.all() else "-"
        ax_led.plot(xvals, shown_vals, color=line_col, lw=2.0, alpha=0.82,
                    ls=line_style, zorder=2)

        statuses = [permutation_p < 0.05,
                    None if pd.isna(pretrend_p) else pretrend_p > 0.05,
                    drop_ok]
        colors = [
            PAL["green"] if statuses[0] else POST,
            PAL["muted"] if statuses[1] is None else (PAL["green"] if statuses[1] else POST),
            PAL["green"] if statuses[2] else POST,
        ]
        for xi, yi, col, is_finite in zip(xvals, shown_vals, colors, finite):
            marker = "o" if is_finite else "D"
            ax_led.scatter(xi, yi, s=120, marker=marker, facecolor=col,
                           edgecolor="white", lw=1.3, zorder=4)
    status_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAL["green"],
               markeredgecolor="white", markeredgewidth=1.1, markersize=9, label="pass"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=POST,
               markeredgecolor="white", markeredgewidth=1.1, markersize=9, label="fail"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PAL["muted"],
               markeredgecolor="white", markeredgewidth=1.1, markersize=8, label="n.a."),
    ]
    design_handles = [
        Line2D([0], [0], color=design_cols[code], lw=2.4,
               ls=(0, (2, 2)) if code == "P4" else "-",
               label=f"{code} {design_names[code]}")
        for code in ["P1", "P2", "P3", "P4"]
    ]
    ax_led.set_xticks(gate_x)
    ax_led.set_xticklabels(gate_labels)
    ax_led.set_xlim(-0.30, 2.28)
    ax_led.set_ylim(-1.22, 1.05)
    ax_led.set_ylabel("Screen margin\n(pass above zero)")
    status_leg = ax_led.legend(handles=status_handles, loc="upper right", ncol=3,
                               handlelength=1.0, columnspacing=0.9,
                               handletextpad=0.35, borderaxespad=0.35)
    ax_led.add_artist(status_leg)
    S.title(ax_led, "Identification screens as signed margins", PAL["ink"])
    S.panel(ax_led, "d")
    S.polish(ax_led, grid_axis="y")
    ax_led_leg.axis("off")
    design_leg = ax_led_leg.legend(handles=design_handles, loc="center", ncol=4,
                                   fontsize=9.0, handlelength=1.7,
                                   columnspacing=0.95, handletextpad=0.45,
                                   borderaxespad=0.0, frameon=True)
    design_leg.get_frame().set_facecolor("white")
    design_leg.get_frame().set_edgecolor(PAL["grid"])
    design_leg.get_frame().set_alpha(0.94)

    S.save(fig, "fig7_identification_stress_test", h_pad=0.12, hspace=0.055)


def main() -> None:
    print("building upgraded quantitative figures ...")
    fig2_idea_space_atlas()
    fig3_multilens_evidence()
    fig4_temporal_rupture()
    fig5_exposure_landscape()
    fig6_decomposition_engine()
    fig7_identification_stress_test()
    print("done.")


if __name__ == "__main__":
    main()
