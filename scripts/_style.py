"""The little that figure scripts share: the palette, the series labels, and the
two loaders.

  * The palette and line styles, because they MUST carry over unchanged between
    Figures 2 and 3.
  * `payload()`, one line, so no script can silently plot a file that is not there.
  * `stimuli()`, because Figure 4 and Supplementary Figure 2 must draw the same
    arrays rather than recompute them.

Figures are written as PDF, the manuscript deliverable. Exactly one PNG is written
alongside -- see `save`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

# A black, B(d0=2) solid blue, B(d0=0) dashed light blue, C orange.
# Blue against orange is the standard color-vision-safe pair; the two blues are
# separated by line style as well as by lightness so the distinction survives
# greyscale printing.
SERIES = {
    "model_a":      dict(label="Model A",
                         color="#000000", linestyle="-",  marker="o"),
    "model_b_d0_2": dict(label="Model B, $d_0 = 2$",
                         color="#1f4e9c", linestyle="-",  marker="s"),
    "model_b_d0_0": dict(label="Model B, $d_0 = 0$",
                         color="#7ba7d7", linestyle="--", marker="^"),
    "model_c":      dict(label=r"Model C",
                         color="#e07b1f", linestyle="-",  marker="D"),
}
ORDER = ["model_a", "model_b_d0_2", "model_b_d0_0", "model_c"]

# Figure 4, Supplementary Figure 2 and the widget all outline the mask, and mark the
# filter envelope against it, so the colors live here: the same element must not
# change color between figures.
MASK_OUTLINE = "#000000"         # black
ENVELOPE_2SIGMA = "#d62aa8"      # magenta
ENVELOPE_3SIGMA = "#2e8b57"      # green
BAND_GREY = "#d9d9d9"            # the 3-sigma band on Figures 2 and 3

# One panel size across Figures 2 and 3, so a reader comparing them is comparing
# the curves and not the aspect ratios. These are the dimensions of the AXES, in
# inches, not of the canvas -- `tight_layout` sizes each figure's margins to its own
# labels, so two nominally identical panels can come out different sizes. `panels()`
# fixes the margins instead and lets the canvas follow.
PANEL_W, PANEL_H = 3.86, 2.63
MARGIN = dict(left=0.74, right=0.16, bottom=0.62, top=0.36, wspace=0.78)


def panels(n=1):
    """A figure whose n panels are each exactly PANEL_W x PANEL_H inches.

    Do not call `tight_layout` on the result: it would undo the fixed margins and
    reintroduce the per-figure variation this exists to remove.
    """
    import matplotlib.pyplot as plt
    m = MARGIN
    fig_w = m["left"] + n * PANEL_W + (n - 1) * m["wspace"] + m["right"]
    fig_h = m["bottom"] + PANEL_H + m["top"]
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=m["left"] / fig_w, right=1 - m["right"] / fig_w,
                        bottom=m["bottom"] / fig_h, top=1 - m["top"] / fig_h,
                        wspace=m["wspace"] / PANEL_W)
    return fig, axes[0]

RC = {
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
}


def payload(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(REPO)} -- run its sweep driver first")
    return json.loads(path.read_text())


def stimuli() -> dict[str, np.ndarray]:
    """Every image array Figure 4 and Supplementary Figure 2 draw. One source, so
    they cannot disagree."""
    path = RESULTS / "stimuli.npz"
    if not path.exists():
        raise SystemExit("missing results/stimuli.npz -- run scripts/make_stimuli.py first")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# PNG is written for Supplementary Figure 2 alone. PDF is the deliverable and the
# form the manuscripts take, but GitHub renders PDF not at all, so the one figure
# README.md embeds needs a raster copy.
PNG_STEMS = {"SuppFig2_model_A"}


def save(fig, stem: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    exts = ["pdf"] + (["png"] if stem in PNG_STEMS else [])
    for ext in exts:
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight")
    print(f"  wrote figures/{stem}." + " and .".join(exts))
