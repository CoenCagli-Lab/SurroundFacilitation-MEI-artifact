"""Dilation sweep. Writes `results/<series>_dilation.json`. Feeds Figure 2.

The headline manipulation. Expand the MEI mask outward and re-optimize the surround
at every expansion. If the reported facilitation is an artifact of a mask drawn
inside the filter envelope, it must vanish once the mask reaches the envelope.

The MEI does not depend on the dilation, so it is optimized ONCE. The mask is
rebuilt at each expansion and the surround re-optimized against it, under BOTH
objectives -- the suppressive surround is a mechanism-specific fingerprint and the
strongest available check that a model is implemented correctly.

Dilation is matched in PIXELS (0, 4, 9, 14, 20, 25). Every configuration here
shares a field of 2.67 deg.

The driver prints each series against the committed values as it finishes, so a run
that has drifted shows up during the 53 minutes rather than after them.

    python3 scripts/sweep_dilation.py [--series model_a ...]
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/featurevis"))

from csartifact.config import Config
from csartifact.mask import mask_radius_px
from csartifact.measure import facilitation_measures
from csartifact.models import build
from csartifact.optimize import (center_only, mask_for, optimize_mei,
                                 optimize_surround)
from _payload import meta, plateau_pct, write

torch.set_num_threads(4)

DILATIONS = (0, 4, 9, 14, 20, 25)

SERIES = {
    "model_a":      Config(model="model_a"),
    "model_b_d0_2": Config(model="model_b"),
    "model_b_d0_0": Config(model="model_b", pool_threshold=0.0),
    "model_c":      Config(model="model_c"),
}

# The committed values, for the live comparison this driver prints. A fresh run
# should land on them: the sweeps are deterministic under their fixed seeds, so a
# move of more than about 10% is an environment or port problem rather than a new
# result, and the driver says so loudly rather than leaving it to be noticed later.
COMMITTED = {
    "model_a":       {"fac_at_9": 1.1082, "sup_at_0": -26.6355, "sup_at_25": -0.0003},
    "model_b_d0_2":  {"fac_at_9": 1.0428, "sup_at_0": -63.7571, "sup_at_25": -45.0651},
    "model_b_d0_0":  {"fac_at_9": -2.8933, "sup_at_0": -69.2517, "sup_at_25": -55.0838},
    "model_c":       {"fac_at_9": 30.7094, "sup_at_0": -58.1491, "sup_at_25": -41.5321},
}


def run(name: str, base: Config) -> dict:
    model = build(base)
    mei, mei_trace = optimize_mei(model, base, return_trace=True)

    dilation_px, mask_radius_sigma = [], []
    curves = {"facilitatory": {"response_pct": [], "drive_pct": []},
              "suppressive": {"response_pct": [], "drive_pct": []}}
    extras, plateaus = [], []

    for px in DILATIONS:
        cfg = dataclasses.replace(base, mask_dilation_px=px)
        mask = mask_for(mei, cfg)
        center = center_only(mei, mask)

        dilation_px.append(px)
        mask_radius_sigma.append(mask_radius_px(mask.numpy().squeeze()) / cfg.sigma_px)

        row_plateau = {"dilation_px": px}
        for objective, key in (("max", "facilitatory"), ("min", "suppressive")):
            composite, trace = optimize_surround(model, mei, mask, cfg,
                                                 objective=objective, return_trace=True)
            fm = facilitation_measures(model, mei, mask, composite)
            curves[key]["response_pct"].append(100.0 * fm["response_ratio"])
            curves[key]["drive_pct"].append(100.0 * fm["drive_ratio"])
            row_plateau[f"{key}_plateau_pct"] = plateau_pct(trace)
        plateaus.append(row_plateau)

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
                     mei_plateau_pct=plateau_pct(mei_trace),
                     plateaus=plateaus,
                     per_dilation=extras),
        "dilation_px": dilation_px,
        "mask_radius_sigma": mask_radius_sigma,
        "facilitatory": curves["facilitatory"],
        "suppressive": curves["suppressive"],
    }


def _flag_large_moves(name: str, data: dict) -> list[str]:
    """Anything more than ~10% from the committed value."""
    if name not in COMMITTED:
        return ["    (no committed values for this series, so nothing to compare)"]
    at = {px: i for i, px in enumerate(data["dilation_px"])}
    got = {
        "fac_at_9": data["facilitatory"]["response_pct"][at[9]],
        "sup_at_0": data["suppressive"]["response_pct"][at[0]],
        "sup_at_25": data["suppressive"]["response_pct"][at[25]],
    }
    notes = []
    for key, before in COMMITTED[name].items():
        now = got[key]
        if abs(before) < 0.01:
            continue
        change = abs(now - before) / abs(before)
        marker = "  <<< DOES NOT MATCH WHAT IS COMMITTED" if change > 0.10 else ""
        notes.append(f"    {key:>10}: {before:+9.4f} -> {now:+9.4f}  "
                     f"({100 * change:+.1f}%){marker}")
    return notes


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
        print(f"  against the committed values:", flush=True)
        for line in _flag_large_moves(name, data):
            print(line, flush=True)
        print(f"  wrote {path.name}  ({(time.perf_counter() - t) / 60:.1f} min)", flush=True)

    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
