"""Stimulus arrays. Writes `results/stimuli.npz`. 

Every image the figures draw, computed once and stored as arrays rather than as JSON.

Per series, six arrays:

    <series>_mei                      the optimized MEI, full field
    <series>_mask                     the soft mask in [0, 1]
    <series>_surround_facilitatory    composite minus the masked center
    <series>_surround_suppressive     same, minimizing objective
    <series>_composite_facilitatory   masked MEI plus facilitatory surround
    <series>_composite_suppressive    masked MEI plus suppressive surround

    python3 scripts/make_stimuli.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/featurevis"))

from csartifact.config import Config
from csartifact.mask import mask_radius_px
from csartifact.models import build
from csartifact.optimize import (center_only, mask_for, optimize_mei,
                                 optimize_surround)
from _payload import RESULTS

# Figure 4's three blocks. Model B here is the PRIMARY d_0 = 2, not the as-published
# variant -- Figure 4 shows the mechanism, and Figures 2 and 3 carry the d_0 = 0
# comparison.
SERIES = {
    "model_a":      Config(model="model_a"),
    "model_b_d0_2": Config(model="model_b"),
    "model_c":      Config(model="model_c"),
}


def main() -> int:
    t0 = time.perf_counter()
    arrays: dict[str, np.ndarray] = {}

    for name, cfg in SERIES.items():
        t = time.perf_counter()
        model = build(cfg)
        mei = optimize_mei(model, cfg)
        mask = mask_for(mei, cfg)
        center = center_only(mei, mask)

        block = {"mei": mei, "mask": mask}
        for objective, key in (("max", "facilitatory"), ("min", "suppressive")):
            composite = optimize_surround(model, mei, mask, cfg, objective=objective)
            block[f"surround_{key}"] = composite - center
            block[f"composite_{key}"] = composite

        for key, tensor in block.items():
            arrays[f"{name}_{key}"] = np.asarray(tensor).squeeze().astype(np.float32)

        radius = mask_radius_px(mask.numpy().squeeze()) / cfg.sigma_px
        print(f"  {name}: 6 arrays, mask {radius:.4f} sigma  "
              f"({time.perf_counter() - t:.0f} s)", flush=True)

    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / "stimuli.npz", **arrays)

    size_kb = (RESULTS / "stimuli.npz").stat().st_size / 1024
    print(f"\nwrote stimuli.npz ({len(arrays)} arrays, {size_kb:.0f} KB) "
          f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
