"""Figure 2 -- Mask dilation sweep, response and drive.

Reads results/<series>_dilation.json. No optimization here; the payloads were
written by scripts/sweep_dilation.py, which takes about an hour.

The headline manipulation: expand the MEI mask outward and re-optimize the surround
at every expansion. If the reported facilitation is an artifact of a mask drawn
inside the filter envelope, it must vanish once the mask reaches the envelope.

The grey band marks where that happens. It is a BAND rather than a line because the
four models reach 3 sigma at slightly different dilations -- Model A's mask is the
largest to begin with (1.932 sigma against about 1.74 for B and C) so it crosses first, at
8.47 px, and Model C last, at 9.74 px. To the left of the band the mask still sits
inside the filter and leakage is available; to the right it does not.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import BAND_GREY, ORDER, RC, SERIES, panels, payload, save

plt.rcParams.update(RC)

data = {s: payload(f"{s}_dilation") for s in ORDER}


def crosses_3sigma(d):
    """Dilation at which this model's mask radius reaches 3 sigma, interpolated."""
    px, rs = d["dilation_px"], d["mask_radius_sigma"]
    for i in range(1, len(px)):
        if rs[i - 1] < 3.0 <= rs[i]:
            return px[i - 1] + (px[i] - px[i - 1]) * (3.0 - rs[i - 1]) / (rs[i] - rs[i - 1])
    return None


crossings = [crosses_3sigma(d) for d in data.values()]
band_lo, band_hi = min(crossings), max(crossings)

fig, axes = panels(2)

for ax, key, ylabel, title in (
    (axes[0], "response_pct", "facilitation, % change in response", "a   response"),
    (axes[1], "drive_pct", "facilitation, % change in drive", "b   drive"),
):
    ax.axvspan(band_lo, band_hi, color=BAND_GREY, zorder=0)

    for s in ORDER:
        st = SERIES[s]
        ax.plot(data[s]["dilation_px"], data[s]["facilitatory"][key],
                color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                label=st["label"], zorder=3)

    ax.set_xlabel("mask dilation (px)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks([0, 4, 9, 14, 20, 25])
    ax.set_xlim(-1, 26)
    ax.axhline(0, color="#999999", linewidth=0.6, zorder=1)
    ax.annotate(r"$3\sigma$", xy=((band_lo + band_hi) / 2, 1.005),
                xycoords=("data", "axes fraction"), color="#666666",
                fontsize=RC["legend.fontsize"],
                ha="center", va="bottom")

axes[0].legend(frameon=False, loc="upper right", borderaxespad=0.2)
save(fig, "Fig2_dilation")

print(f"  3-sigma band spans {band_lo:.2f} to {band_hi:.2f} px")
print("  facilitation at the 9 px expansion:")
for s in ORDER:
    i = data[s]["dilation_px"].index(9)
    print(f"    {SERIES[s]['label']:<38} {data[s]['facilitatory']['response_pct'][i]:+9.4f}%")
