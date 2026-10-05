"""Dense Model A dilation sweep for the widget. Writes
`results/model_a_dilation_dense.json`.

WIDGET ONLY. This payload feeds no printed figure and no printed number, and its
`meta` block says so. It is the one exception to the rule that a file enters this
repository only if it feeds a printed figure or a printed number.

ANALYTIC ONLY -- NOTHING IS RE-SIMULATED. There is no optimizer here and no surround
ascent. At each slider position the sweep evaluates `mask.analytic_headroom`, the
closed-form bound on how much drive a stimulus confined outside the mask can
contribute. The MEI is not re-optimized either: it is read from `results/stimuli.npz`,
the same array the figures draw, so the widget cannot disagree with them about
what the mask is drawn around. The whole run is a second.

THE WIDGET'S CURVE IS NOT FIGURE 2's CURVE. Figure 2 plots what the optimizer
achieved (27.31% undilated); this plots the bound it was working against (28.71%).

    python3 scripts/sweep_dilation_dense.py
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/featurevis"))

from csartifact.config import Config
from csartifact.mask import analytic_headroom, mask_radius_px
from csartifact.models import build
from csartifact.optimize import center_only, mask_for
from _payload import RESULTS, meta, write

torch.set_num_threads(4)

DILATIONS = list(range(-4, 26))

def main() -> int:
    t0 = time.perf_counter()
    cfg = Config(model="model_a")
    model = build(cfg)
    f = model.center_filter

    with np.load(RESULTS / "stimuli.npz") as z:
        mei = torch.tensor(z["model_a_mei"], dtype=torch.float32)[None, None]

    dilation_px, mask_radius_sigma = [], []
    drive_ratio_pct, response_pct, center_drive = [], [], []

    for px in DILATIONS:
        c = dataclasses.replace(cfg, mask_dilation_px=px)
        mask = mask_for(mei, c)
        m = mask.numpy().squeeze()

        bound = analytic_headroom(f, m, c.contrast_center, c.contrast_surround)
        d_c = float(model.drive(center_only(mei, mask)))

        dilation_px.append(px)
        mask_radius_sigma.append(mask_radius_px(m) / cfg.sigma_px)
        drive_ratio_pct.append(100.0 * bound)
        center_drive.append(d_c)
        response_pct.append(100.0 * bound * d_c / (d_c + 1.0))

    data = {
        "meta": meta("model_a_dense", cfg,
                     widget_only=True,
                     feeds_no_printed_output=True,
                     method="analytic_headroom; no optimizer, no surround ascent",
                     mei_source="results/stimuli.npz :: model_a_mei",
                     anchor="drive_ratio_pct at dilation 0 == the analytic bound "
                            "for the undilated mask == 28.7126",
                     note="the analytic BOUND, not the optimizer result; Figure 2 "
                          "plots what the optimizer achieved and is a different curve"),
        "dilation_px": dilation_px,
        "mask_radius_sigma": mask_radius_sigma,
        "drive_ratio_pct": drive_ratio_pct,
        "response_pct": response_pct,
        "center_drive": center_drive,
    }
    path = write("model_a_dilation_dense", data)

    at0 = dilation_px.index(0)
    print(f"  {len(dilation_px)} points, {DILATIONS[0]} to {DILATIONS[-1]} px")
    print(f"  {'px':>4} {'r/sigma':>8} {'drive %':>9} {'response %':>11} {'d_c':>8}")
    for i, px in enumerate(dilation_px):
        if px in (-4, -3, -2, -1, 0, 4, 9, 14, 20, 25):
            print(f"  {px:>4} {mask_radius_sigma[i]:>8.4f} {drive_ratio_pct[i]:>9.4f} "
                  f"{response_pct[i]:>11.4f} {center_drive[i]:>8.3f}")
    print(f"\n  anchor: drive at dilation 0 = {drive_ratio_pct[at0]:.4f}% "
          f"(the bound for the undilated mask = 28.7126%)")
    print(f"          response           = {response_pct[at0]:.4f}% "
          f"(28.7126 x {center_drive[at0]:.4f}/{center_drive[at0] + 1:.4f} = "
          f"{28.7126 * center_drive[at0] / (center_drive[at0] + 1):.4f}%)")
    print(f"  wrote {path.name}  ({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
