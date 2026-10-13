"""Dilation sweep. Writes `results/<series>_dilation.json`. Feeds Figure 2.

The headline manipulation. Expand the MEI mask outward and re-optimize the surround
at every expansion. 

The MEI does not depend on the dilation, so it is optimized ONCE. The mask is
rebuilt at each expansion and the surround re-optimized against it, under BOTH
objectives i.e. max facilitation and max suppression.

Dilation is matched in PIXELS (0, 4, 9, 14, 20, 25). Every configuration here
assumes a field of 2.67 deg.

    python3 scripts/sweep_dilation.py [--series model_a ...]
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/featurevis"))

from csartifact.config import Config
from csartifact.mask import mask_radius_px
from csartifact.measure import facilitation_measures
from csartifact.models import build
from csartifact.optimize import (center_only, mask_for, optimize_mei,
                                 optimize_surround)
from _payload import meta, write

DILATIONS = (0, 4, 9, 14, 20, 25)

SERIES = {
    "model_a":      Config(model="model_a"),
    "model_b_d0_2": Config(model="model_b"),
    "model_b_d0_0": Config(model="model_b", pool_threshold=0.0),
    "model_c":      Config(model="model_c"),
}


def run(name: str, base: Config) -> dict:
    model = build(base)
    mei = optimize_mei(model, base)

    dilation_px, mask_radius_sigma = [], []
    curves = {"facilitatory": {"response_pct": [], "drive_pct": []},
              "suppressive": {"response_pct": [], "drive_pct": []}}
    extras = []

    for px in DILATIONS:
        cfg = dataclasses.replace(base, mask_dilation_px=px)
        mask = mask_for(mei, cfg)
        center = center_only(mei, mask)

        dilation_px.append(px)
        mask_radius_sigma.append(mask_radius_px(mask.numpy().squeeze()) / cfg.sigma_px)

        for objective, key in (("max", "facilitatory"), ("min", "suppressive")):
            composite = optimize_surround(model, mei, mask, cfg, objective=objective)
            fm = facilitation_measures(model, mei, mask, composite)
            curves[key]["response_pct"].append(100.0 * fm["response_ratio"])
            curves[key]["drive_pct"].append(100.0 * fm["drive_ratio"])

        extras.append({
            "dilation_px": px,
            "ybar_center": float(model.pool_mean(center)) if hasattr(model, "pool_mean") else 0.0,
            "weight_inside_mask_fraction": (
                float(model.weight_inside_mask_fraction(mask))
                if hasattr(model, "weight_inside_mask_fraction") else None),
        })

        print(f"    {px:>3} px  r/sigma {mask_radius_sigma[-1]:5.3f}  "
              f"fac {curves['facilitatory']['response_pct'][-1]:+9.4f}%  "
              f"sup {curves['suppressive']['response_pct'][-1]:+9.4f}%", flush=True)

    return {
        "meta": meta(name, base,
                     dilations_px=list(DILATIONS),
                     per_dilation=extras),
        "dilation_px": dilation_px,
        "mask_radius_sigma": mask_radius_sigma,
        "facilitatory": curves["facilitatory"],
        "suppressive": curves["suppressive"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", default=list(SERIES), choices=list(SERIES))
    args = ap.parse_args()

    t0 = time.perf_counter()
    for name in args.series:
        t = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        data = run(name, SERIES[name])
        path = write(f"{name}_dilation", data)
        print(f"  wrote {path.name}  ({(time.perf_counter() - t) / 60:.1f} min)", flush=True)

    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
