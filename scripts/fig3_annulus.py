"""Figure 3 -- Annulus sweep, response and drive.

Reads results/<series>_annulus.json. No optimization here.

The leakage argument with no optimizer in it. Take the facilitatory composite and
blank its central disk, sweeping the radius. Nothing is re-optimized and nothing is
renormalized, so this is a pure forward pass and carries no optimizer assumptions.

The headline: blank the ENTIRE mask -- the dotted lines -- and every model still
fires several times above baseline on 13-21% of the intact drive, because
the "surround" pixels just outside the mask are still inside the filter. Response
reaches baseline between 1.25 and 1.5 mask radii, i.e. just inside 3 sigma, which
agrees with the dilation sweep about where the filter actually ends.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import BAND_GREY, ORDER, RC, SERIES, panels, payload, save

plt.rcParams.update(RC)

data = {s: payload(f"{s}_annulus") for s in ORDER}

# The abscissa is the blanked-disk radius in sigma, so the models' own mask radii
# differ and the "original mask" marker is a band, not a line: Model A's mask is
# 1.932 sigma, the others about 1.74.
mask_radii = [d["meta"]["mask_radius_sigma"] for d in data.values()]

fig, axes = panels(2)

for ax, key, ylabel, title in (
    (axes[0], "response_rel_blank", r"response, $\times$ blank-screen baseline", "a   response"),
    (axes[1], "drive_pct_of_full", "drive, % of the intact composite", "b   drive"),
):
    for s in ORDER:
        st = SERIES[s]
        ax.plot(data[s]["radii_sigma"], data[s][key],
                color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                label=st["label"], zorder=3)

    # dotted lines at the original mask radii, grey band at 3 sigma
    for r in (min(mask_radii), max(mask_radii)):
        ax.axvline(r, color="#666666", linestyle=":", linewidth=0.8, zorder=1)
    ax.axvspan(2.9, 3.1, color=BAND_GREY, zorder=0)

    ax.set_xlabel(r"blanked-disk radius ($\sigma$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(-0.2, 5.2)
    ax.axhline(1.0 if key == "response_rel_blank" else 0.0,
               color="#999999", linewidth=0.6, zorder=1)

axes[0].legend(frameon=False, loc="upper right")

# Both markers label the vertical lines they sit over, from above the axes.
for ax in axes:
    ax.annotate("original\nmask", xy=((min(mask_radii) + max(mask_radii)) / 2, 1.005),
                xycoords=("data", "axes fraction"),
                fontsize=RC["legend.fontsize"], color="#666666",
                ha="center", va="bottom")
    ax.annotate(r"$3\sigma$", xy=(3.0, 1.005), xycoords=("data", "axes fraction"),
                fontsize=RC["legend.fontsize"], color="#666666",
                ha="center", va="bottom")

save(fig, "Fig3_annulus")

print("  with the entire mask blanked (radius = 1.0 mask radii):")
for s in ORDER:
    i = data[s]["meta"]["radii_mult_of_mask"].index(1.0)
    print(f"    {SERIES[s]['label']:<38} {data[s]['response_rel_blank'][i]:6.2f}x baseline, "
          f"{data[s]['drive_pct_of_full'][i]:5.1f}% of intact drive")
