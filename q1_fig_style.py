"""Shared Nature-style visual system for the CreativeAI Scientometrics figures.

The goal is a coherent figure family: large readable type, restrained but vivid
color, high-resolution output, and reusable helpers for density fields, rainclouds,
ridge-like traces, and sensitivity bands.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Colorblind-aware palette with a single editorial voice across all figures.
PAL = {
    "ink": "#17212B",
    "slate": "#3E5060",
    "muted": "#8795A1",
    "grid": "#E7EDF1",
    "paper": "#FFFFFF",
    "wash": "#F7FAFC",
    "blue": "#0E6FAE",
    "sky": "#69B7DC",
    "teal": "#1B9E91",
    "green": "#0E9272",
    "gold": "#D99A22",
    "sand": "#F0C36A",
    "vermil": "#D65F37",
    "rose": "#B83D57",
    "purple": "#7E62A3",
    "violet": "#A283BF",
}

PRE = PAL["blue"]
POST = PAL["vermil"]
LENSES = [PAL["blue"], PAL["teal"], PAL["purple"]]


def _preferred_font() -> str:
    for name in ("Arial", "Helvetica", "Aptos", "Liberation Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


def set_theme() -> None:
    fam = _preferred_font()
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 450,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": [fam, "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "axes.labelpad": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.edgecolor": PAL["ink"],
        "axes.linewidth": 0.9,
        "axes.labelcolor": PAL["ink"],
        "text.color": PAL["ink"],
        "xtick.color": PAL["ink"],
        "ytick.color": PAL["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.pad": 7,
        "ytick.major.pad": 7,
        "axes.grid": True,
        "grid.color": PAL["grid"],
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": PAL["paper"],
        "axes.facecolor": PAL["paper"],
        "lines.antialiased": True,
        "patch.antialiased": True,
    })


def panel(ax, letter: str, dx: float = -0.02, dy: float = 1.07) -> None:
    current = ax.get_title(loc="left") or ax.get_title()
    prefix = f"({letter})"
    if current and not current.startswith(prefix):
        ax.set_title(f"{prefix} {current}", loc="left", fontsize=14,
                     fontweight="bold", color=PAL["ink"], pad=15)
    elif not current:
        ax.set_title(prefix, loc="left", fontsize=14,
                     fontweight="bold", color=PAL["ink"], pad=15)


def title(ax, text: str, color: str | None = None, size: int = 14) -> None:
    text = text.replace(":", " -")
    ax.set_title(text, loc="left", fontsize=size, fontweight="bold",
                 color=PAL["ink"], pad=15)


def note(ax, text: str, xy=(0.02, 0.98), color: str | None = None,
         ha: str = "left", va: str = "top", size: int = 11) -> None:
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=size, color=color or PAL["slate"], fontweight="bold")


def polish(ax, grid_axis: str = "y") -> None:
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=PAL["grid"], lw=0.8)


def blend_with_white(color: str, frac: float = 0.65) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(color))
    return tuple(rgb * (1 - frac) + np.ones(3) * frac)


def cmap_from(color: str, alpha_max: float = 0.82, name: str = "custom"):
    r, g, b = to_rgb(color)
    return LinearSegmentedColormap.from_list(
        name, [(r, g, b, 0.0), (r, g, b, alpha_max)]
    )


def signed_cmap():
    return LinearSegmentedColormap.from_list(
        "signed_blue_vermil",
        [PAL["blue"], "#F8FAFC", PAL["vermil"]],
        N=256,
    )


def half_violin(ax, xpos, data, color, side="right", width=0.34, alpha=0.32):
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        return
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    lo, hi = data.min(), data.max()
    pad = 0.08 * (hi - lo + 1e-9)
    ys = np.linspace(lo - pad, hi + pad, 240)
    dens = kde(ys)
    dens = dens / dens.max() * width
    sign = 1 if side == "right" else -1
    ax.fill_betweenx(ys, xpos, xpos + sign * dens, color=color, alpha=alpha, lw=0)
    ax.plot(xpos + sign * dens, ys, color=color, lw=1.7)
    ax.plot([xpos, xpos], [np.percentile(data, 25), np.percentile(data, 75)],
            color=color, lw=5, solid_capstyle="round", zorder=5)
    ax.scatter([xpos], [np.median(data)], s=44, color="white",
               edgecolor=color, lw=1.6, zorder=6)


def bees(ax, xpos, data, color, width=0.16, size=10, alpha=0.28, seed=7):
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    rng = np.random.default_rng(seed)
    if data.size > 900:
        data = rng.choice(data, 900, replace=False)
    jitter = rng.normal(0, width / 2.2, size=data.size)
    jitter = np.clip(jitter, -width, width)
    ax.scatter(np.full(data.size, xpos) + jitter, data, s=size, color=color,
               alpha=alpha, ec="none", zorder=2)


def density_grid(x, y, gridsize=180, margin=0.07, sample=9000, seed=13):
    """Return a shared-grid Gaussian KDE for an x/y point cloud."""
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(x.size), sample, replace=False)
        x, y = x[idx], y[idx]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xr, yr = xmax - xmin, ymax - ymin
    xx, yy = np.mgrid[
        xmin - margin * xr:xmax + margin * xr:complex(gridsize),
        ymin - margin * yr:ymax + margin * yr:complex(gridsize),
    ]
    vals = gaussian_kde(np.vstack([x, y]))(np.vstack([xx.ravel(), yy.ravel()]))
    zz = vals.reshape(xx.shape)
    return xx, yy, zz


def contour_density(ax, xx, yy, zz, color, levels=9, filled=True):
    norm = Normalize(vmin=np.nanmin(zz), vmax=np.nanmax(zz))
    if filled:
        ax.contourf(xx, yy, zz, levels=levels, cmap=cmap_from(color),
                    antialiased=True, zorder=1)
    ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=1.15,
               alpha=0.78, zorder=2)
    return norm


def save(fig, name: str, *, w_pad: float = 0.24, h_pad: float = 0.24,
         wspace: float = 0.12, hspace: float = 0.14) -> None:
    if fig.get_constrained_layout():
        fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad,
                                        wspace=wspace, hspace=hspace)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", pad_inches=0.035, dpi=450)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"  saved figures/{name}.png and figures/{name}.pdf")
