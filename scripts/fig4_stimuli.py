"""Figure 4 -- MEI and mask, both surrounds, both composites, per model.

Reads results/stimuli.npz. No optimization here.

Three blocks, one per model, each 3 columns x 2 sub-rows:

    row 1   MEI (mask outlined)   facilitatory surround   facilitatory composite
    row 2   "MEI as above"        suppressive surround    suppressive composite

FIFTEEN IMAGES ON ONE SHARED LUMINANCE SCALE, +/-1.263, which is the measured
extent across all of them. The shared scale is the point: on it, Model C's panels
read as visibly lower-contrast than Model B's, and that is real -- the optimizer
distributes the same 0.10 RMS surround budget differently between the models. 

The mask is outlined on the MEI and on the surrounds, but NOT on the composites --
the composite is what the model actually sees.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import MASK_OUTLINE, RC, save, stimuli

plt.rcParams.update(RC)

# One luminance scale across all fifteen panels, so the three models are comparable
# by eye. Measured by make_stimuli.py, which prints it at the end of its run.
SHARED_EXTENT = 1.263
IMG_PX = 93
BLOCKS = [("model_a", "a   Model A"), ("model_b_d0_2", "b   Model B"),
          ("model_c", "c   Model C")]

arrays = stimuli()
grid = np.linspace(-IMG_PX / 2, IMG_PX / 2, IMG_PX)
extent = [-IMG_PX / 2, IMG_PX / 2, -IMG_PX / 2, IMG_PX / 2]

fig, axes = plt.subplots(6, 3, figsize=(6.6, 13.2))


def show(ax, img, title, mask=None):
    ax.imshow(img, cmap="gray", vmin=-SHARED_EXTENT, vmax=SHARED_EXTENT,
              extent=extent, origin="lower")
    if mask is not None:
        ax.contour(grid, grid, mask, levels=[0.5], colors=MASK_OUTLINE, linewidths=1.0)
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines:
        ax.spines[side].set_visible(False)


for block, (series, label) in enumerate(BLOCKS):
    mask = arrays[f"{series}_mask"]
    r0, r1 = 2 * block, 2 * block + 1

    show(axes[r0][0], arrays[f"{series}_mei"], f"{label}    MEI", mask)
    show(axes[r0][1], arrays[f"{series}_surround_facilitatory"],
         "facilitatory surround", mask)
    show(axes[r0][2], arrays[f"{series}_composite_facilitatory"],
         "MEI + facilitatory surround")

    axes[r1][0].axis("off")
    axes[r1][0].text(0.5, 0.5, "MEI as above\n(unchanged)", ha="center", va="center",
                     fontsize=7.5, color="#555555", transform=axes[r1][0].transAxes)
    show(axes[r1][1], arrays[f"{series}_surround_suppressive"],
         "suppressive surround", mask)
    show(axes[r1][2], arrays[f"{series}_composite_suppressive"],
         "MEI + suppressive surround")

fig.subplots_adjust(hspace=0.18, wspace=0.04)
cbar = fig.colorbar(axes[0][0].images[0], ax=axes, orientation="horizontal",
                    fraction=0.020, pad=0.025, aspect=50)
cbar.set_label(f"shared luminance scale, $\\pm${SHARED_EXTENT}", fontsize=7.5)
cbar.ax.tick_params(labelsize=7)

save(fig, "Fig4_stimuli")

print(f"  15 images on one shared scale, +/-{SHARED_EXTENT}")
for series, label in BLOCKS:
    peak = max(abs(arrays[f"{series}_composite_facilitatory"]).max(),
               abs(arrays[f"{series}_composite_suppressive"]).max())
    print(f"    {label.split()[-1]:<8} peak |value| {peak:.4f}")
