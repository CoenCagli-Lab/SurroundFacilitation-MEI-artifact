"""Annulus sweep. Writes `results/<series>_annulus.json`. Feeds Figure 3.

Take the composite at the default settings -- undilated, primary threshold,
facilitatory objective -- and replace the central disk with mean luminance, sweeping
the disk radius in multiples of the mask radius.

**Nothing is re-optimized and nothing is renormalized after blanking.**
The manipulation is a pure forward pass.

The headline: blank the ENTIRE mask and every model still fires several times above
baseline on 13-21% of the intact drive, because the "surround" pixels adjacent to
the mask are still inside the filter.

Edge treatment is HARD, `keep = (dist >= r)`. No raised-cosine alternative is
computed (DEVIATIONS.md, D6).

    python3 scripts/sweep_annulus.py [--series model_a ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/featurevis"))

from csartifact.config import Config
from csartifact.mask import mask_radius_px
from csartifact.measure import (ANNULUS_RADII_MULT, blank_center,
                                radial_distance)
from csartifact.models import build
from csartifact.optimize import mask_for, optimize_mei, optimize_surround
from _payload import meta, plateau_pct, write

torch.set_num_threads(4)

SERIES = {
    "model_a":      Config(model="model_a"),
    "model_b_d0_2": Config(model="model_b"),
    "model_b_d0_0": Config(model="model_b", pool_threshold=0.0),
    "model_c":      Config(model="model_c"),
}


def run(name: str, cfg: Config) -> dict:
    model = build(cfg)

    mei, mei_trace = optimize_mei(model, cfg, return_trace=True)
    mask = mask_for(mei, cfg)
    composite, sur_trace = optimize_surround(model, mei, mask, cfg,
                                             objective="max", return_trace=True)

    dist = radial_distance(cfg.img_px)
    r_mask = mask_radius_px(mask.numpy().squeeze())
    baseline = float(model.response(torch.zeros_like(composite)))
    drive_intact = float(model.drive(composite))

    radii_px, response_rel_blank, drive_pct_of_full, radii_sigma = [], [], [], []
    for mult in ANNULUS_RADII_MULT:
        r = mult * r_mask
        x = blank_center(composite, dist, r)
        radii_px.append(r)
        radii_sigma.append(r / cfg.sigma_px)
        response_rel_blank.append(float(model.response(x)) / baseline)
        drive_pct_of_full.append(100.0 * float(model.drive(x)) / drive_intact)

    return {
        "meta": meta(name, cfg,
                     objective="facilitatory",
                     edge="hard",
                     renormalized=False,
                     mask_radius_px=r_mask,
                     mask_radius_sigma=r_mask / cfg.sigma_px,
                     blank_baseline_response=baseline,
                     drive_intact=drive_intact,
                     radii_mult_of_mask=list(ANNULUS_RADII_MULT),
                     mei_plateau_pct=plateau_pct(mei_trace),
                     surround_plateau_pct=plateau_pct(sur_trace)),
        "radii_px": radii_px,
        "radii_sigma": radii_sigma,
        "response_rel_blank": response_rel_blank,
        "drive_pct_of_full": drive_pct_of_full,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", default=list(SERIES), choices=list(SERIES))
    args = ap.parse_args()

    t0 = time.perf_counter()
    for name in args.series:
        t = time.perf_counter()
        data = run(name, SERIES[name])
        path = write(f"{name}_annulus", data)
        m = data["meta"]
        print(f"\n=== {name}  ({time.perf_counter() - t:.0f} s, mask "
              f"{m['mask_radius_sigma']:.3f} sigma, baseline {m['blank_baseline_response']:.4f}, "
              f"intact drive {m['drive_intact']:.4f}) ===", flush=True)
        print(f"  {'r/mask':>7} {'r/sigma':>8} {'resp/blank':>11} {'drive % of full':>16}",
              flush=True)
        for mult, rs, rr, dp in zip(ANNULUS_RADII_MULT, data["radii_sigma"],
                                    data["response_rel_blank"], data["drive_pct_of_full"]):
            print(f"  {mult:>7.4g} {rs:>8.3f} {rr:>11.3f} {dp:>15.2f}%", flush=True)
        print(f"  wrote {path.name}", flush=True)

    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
