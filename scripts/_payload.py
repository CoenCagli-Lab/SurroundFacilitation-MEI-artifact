"""Writing payloads to `results/`.

Each payload carries a `meta` block naming the series, the resolved optimizer
settings, the geometry and the time it was written.

See the README on verification.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def meta(series: str, cfg, **extra) -> dict:
    return {
        "series": series,
        "model": cfg.model,
        "optimizer": {
            "step_size": cfg.opt_step_size,
            "mei_iterations": cfg.opt_mei_iters,
            "surround_iterations": cfg.opt_surround_iters,
            "pool_threshold": cfg.opt_pool_threshold,
            "gradient_blur_sigma": cfg.grad_blur_sigma,
        },
        "geometry": {
            "img_px": cfg.img_px,
            "deg_per_image": cfg.deg_per_image,
            "sigma_px": cfg.sigma_px,
            "ppd": cfg.ppd,
        },
        "beta": cfg.beta if cfg.model == "model_c" else None,
        "seed": cfg.seed,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


def write(name: str, data: dict) -> Path:
    """Write `results/<name>.json`. Scalars and curves only -- no image arrays."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    return path


def plateau_pct(fevals) -> float:
    """Change over the last 10% of the ascent, as a percentage of the final value.

    Every optimization result on a nonlinear model gets this checked before it is
    believed; a curve still climbing at the last iteration has not converged.
    """
    f = np.asarray(fevals, dtype=float)
    k = max(1, int(len(f) * 0.1))
    return float((f[-1] - f[-k]) / abs(f[-1]) * 100.0)
