"""Drive, response, facilitation, and the annulus stimulus.

Two measures:

  response_ratio   R(MEI+surround)/R(MEI) - 1. What the paper reports, and what an
                   experimenter measures. NOT comparable across models: Models B
                   and C are measured through a divisively normalized response and
                   Model A is not, so part of any gap between them is the
                   normalization.

  drive_ratio      <f_i, composite>/<f_i, MEI-masked> - 1. The leakage in terms of
                   filter drive. Linear, unnormalized, and
                   identical in meaning for all three models because they share a
                   center filter. 

The two are tied together by the ELU baseline offset:

    response_facilitation == drive_ratio * d_mei / (d_mei + 1)
"""
from __future__ import annotations

import numpy as np
import torch

# Annulus sweep: blanked-disk radii as multiples of the mask radius.
# r = 0 is the unmodified stimulus and is the reference every other point is
# expressed as a fraction of, so the sweep must start there.
ANNULUS_RADII_MULT: tuple[float, ...] = (
    0.0, 0.25, 0.5, 2.0 / 3.0, 0.8, 1.0, 1.25, 1.5, 2.0, 4.0
)


def facilitation_measures(model, mei: torch.Tensor, mask: torch.Tensor,
                          composite: torch.Tensor) -> dict:
    """All per-run quantities a sweep needs, from one composite."""
    center = mask * mei
    r_center = float(model.response(center))
    r_comp = float(model.response(composite))
    d_center = float(model.drive(center))
    d_comp = float(model.drive(composite))

    out = {
        "response_center": r_center,
        "response_composite": r_comp,
        "response_ratio": r_comp / r_center - 1.0,
        "drive_center": d_center,
        "drive_composite": d_comp,
        "drive_ratio": d_comp / d_center - 1.0 if d_center != 0 else float("nan"),
    }
    if hasattr(model, "pool_mean"):
        out["ybar_center"] = float(model.pool_mean(center))
        out["ybar_composite"] = float(model.pool_mean(composite))
    if hasattr(model, "gain"):
        out["gain_center"] = float(model.gain(center))
        out["gain_composite"] = float(model.gain(composite))
    return out


def radial_distance(img_px: int) -> np.ndarray:
    """Distance of every pixel from the image center, in pixels."""
    center = img_px / 2.0 - 0.5
    yy, xx = np.mgrid[0:img_px, 0:img_px]
    return np.sqrt((yy - center) ** 2 + (xx - center) ** 2)


def annulus_keep(dist: np.ndarray, radius_px: float) -> np.ndarray:
    """Retention weights for a stimulus with its central disk blanked.

    The manipulation is a pure forward pass: nothing is re-optimized and nothing
    is renormalized after blanking, so it carries no optimizer assumptions at all.

    HARD edge, `keep = (dist >= r)`. 
    """
    if radius_px <= 0.0:
        return np.ones_like(dist)
    return (dist >= radius_px).astype(np.float64)


def blank_center(stimulus: torch.Tensor, dist: np.ndarray,
                 radius_px: float) -> torch.Tensor:
    """Apply `annulus_keep` to a stimulus tensor."""
    keep = annulus_keep(dist, radius_px)
    return stimulus * torch.tensor(keep, dtype=torch.float32)[None, None]
