"""Supplementary Figure 2 -- Geometry, both surrounds, both composites. Model A only.

Also the Letter's Figure 2, and panel (a) alone is the Supplement's Figure 1.

Not a recomputation: it draws the same arrays, through `_style.stimuli()`, that
fig4_stimuli.py draws for Figure 4, so the Letter and the Supplement cannot drift
apart. If the stimuli are ever regenerated, the figures move together or not at all.

FIVE image panels:

    (a)  the MEI at twice the linear size of the others, carrying three contours --
         the mask black solid, the 2 sigma envelope magenta dotted, the 3 sigma
         envelope green dashed.
    (b)  facilitatory "surround", and the composite it forms with the MEI.
    (c)  suppressive "surround", and the composite.

"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csartifact.config import Config
from _style import ENVELOPE_2SIGMA, ENVELOPE_3SIGMA, MASK_OUTLINE, RC, save, stimuli

plt.rcParams.update(RC)

# Image size and filter sigma come from the same settings the simulations used.
IMG_PX, SIGMA_PX = Config().img_px, Config().sigma_px
SHARED_EXTENT = 1.263            # the same scale Figure 4 uses

ENV_2SIGMA = ENVELOPE_2SIGMA     # magenta, dotted here
ENV_3SIGMA = ENVELOPE_3SIGMA     # green, dashed here

arrays = stimuli()
mei = arrays["model_a_mei"]
mask = arrays["model_a_mask"]
surround_max = arrays["model_a_surround_facilitatory"]
surround_min = arrays["model_a_surround_suppressive"]
composite_max = arrays["model_a_composite_facilitatory"]
composite_min = arrays["model_a_composite_suppressive"]

mask_radius_sigma = float(np.sqrt((mask > 0.5).sum() / np.pi)) / SIGMA_PX

grid = np.linspace(-IMG_PX / 2, IMG_PX / 2, IMG_PX)
extent = [-IMG_PX / 2, IMG_PX / 2, -IMG_PX / 2, IMG_PX / 2]

# Panel (a) spans two rows and two columns so it is drawn at twice the linear size
# of the four on the right. The extra bottom row carries its inline legend.
fig = plt.figure(figsize=(10.2, 5.4))
gs = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 0.22],
              hspace=0.22, wspace=0.06)

ax_a = fig.add_subplot(gs[0:2, 0:2])
ax_legend = fig.add_subplot(gs[2, 0:2])
ax_b_sur = fig.add_subplot(gs[0, 2])
ax_b_comp = fig.add_subplot(gs[0, 3])
ax_c_sur = fig.add_subplot(gs[1, 2])
ax_c_comp = fig.add_subplot(gs[1, 3])


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines:
        ax.spines[side].set_visible(False)


# ---- (a) geometry, at twice the linear size ------------------------------
ax_a.imshow(mei, cmap="gray", vmin=-abs(mei).max(), vmax=abs(mei).max(),
            extent=extent, origin="lower")
ax_a.contour(grid, grid, mask, levels=[0.5], colors=MASK_OUTLINE, linewidths=1.8,
             linestyles="solid")
ax_a.add_patch(plt.Circle((0, 0), 2.0 * SIGMA_PX, fill=False, color=ENV_2SIGMA,
                          linewidth=2.0, linestyle=":"))
ax_a.add_patch(plt.Circle((0, 0), 3.0 * SIGMA_PX, fill=False, color=ENV_3SIGMA,
                          linewidth=2.0, linestyle="--"))
ax_a.set_title("a   MEI", loc="left", fontweight="bold")
bare(ax_a)

# inline legend, beneath (a) rather than floating over the image
ax_legend.axis("off")
handles = [
    plt.Line2D([], [], color=MASK_OUTLINE, linewidth=2.8, linestyle="solid",
               label=f"MEI mask  ({mask_radius_sigma:.2f}$\\sigma$)"),
    plt.Line2D([], [], color=ENV_2SIGMA, linewidth=2.8, linestyle=":",
               label=r"$2\sigma$ envelope"),
    plt.Line2D([], [], color=ENV_3SIGMA, linewidth=2.8, linestyle="--",
               label=r"$3\sigma$ envelope"),
]
ax_legend.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                 fontsize=8, handlelength=2.4, columnspacing=1.6,
                 borderaxespad=0.0)

# ---- (b) and (c) surrounds and composites, on Figure 4's shared scale ----
for ax, img, title, outline in (
    (ax_b_sur, surround_max, 'b   facilitatory "surround"', True),
    (ax_b_comp, composite_max, 'MEI + facilitatory "surround"', False),
    (ax_c_sur, surround_min, 'c   suppressive "surround"', True),
    (ax_c_comp, composite_min, 'MEI + suppressive "surround"', False),
):
    ax.imshow(img, cmap="gray", vmin=-SHARED_EXTENT, vmax=SHARED_EXTENT,
              extent=extent, origin="lower")
    if outline:
        ax.contour(grid, grid, mask, levels=[0.5], colors=MASK_OUTLINE, linewidths=1.0)
    ax.set_title(title, loc="left", fontweight="bold" if title[1] == " " else "normal",
                 fontsize=8.5)
    bare(ax)

save(fig, "SuppFig2_model_A")

# The arrays are shared with Figure 4, so this is a structural fact about Model A rather
# than a coincidence of two separate optimizations.
r = float(np.corrcoef(surround_max.ravel(), surround_min.ravel())[0, 1])
print(f"  5 panels + inline legend; mask {mask_radius_sigma:.4f} sigma "
      f"against a 2 sigma envelope")
print(f"  corr(facilitatory, suppressive) = {r:+.6f}   (exactly antiphase in Model A)")
