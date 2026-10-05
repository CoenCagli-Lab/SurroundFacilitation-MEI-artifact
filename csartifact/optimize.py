"""MEI and surround optimization.

Reimplemented from their `create_mei` / `create_surround`, with every parameter
routed through `Config`. featurevis supplies `GaussianBlur`, `ChangeStd`,
`ClipRange` and `Compose`; it does NOT supply the center/surround machinery --
their analysis code imports `ops.ChangeSurroundStd`, which does not exist in the
public cajal/featurevis. The classes below fill that gap.

`grad_blur_sigma` is fixed at 1.0, their mouse value, everywhere. It is not a
swept parameter: its effect on mask size and on the measured artifact is
non-monotonic, so varying it explains nothing and only adds noise. Setting it to 0
disables the blur.
"""
from __future__ import annotations

import numpy as np
import torch
from featurevis import ops as fv_ops
from featurevis import utils as fv_utils
from featurevis.core import gradient_ascent

from .config import Config
from .mask import create_mask


# ---- operations ---------------------------------------------------------

class ZeroMean:
    """Subtract the mean. Their MEI step rescales to mean 0, std `contrast`."""

    @fv_utils.varargin
    def __call__(self, x, **kwargs):
        return x - x.mean()


class ClipRange:
    """Pixel clipping in z-scored input space.

    Their `create_mei` and `create_surround` both apply ClipRange(-1.7876, 2.1919)
    every iteration (DEVIATIONS.md D2).
    """

    def __init__(self, low: float = -1.7876, high: float = 2.1919):
        self.low, self.high = low, high

    @fv_utils.varargin
    def __call__(self, x, **kwargs):
        return torch.clamp(x, self.low, self.high)


class MaskedComposite:
    """transform: hold the MEI fixed inside the mask, optimize outside.

        composite = mask * mei + (1 - mask) * x

    Differentiable, so gradients flow to the surround pixels.
    """

    def __init__(self, mei: torch.Tensor, mask: torch.Tensor):
        self.mei, self.mask = mei, mask

    @fv_utils.varargin
    def __call__(self, x, **kwargs):
        return self.mask * self.mei + (1.0 - self.mask) * x


class SurroundGradient:
    """gradient_f for the surround step: blur, and negate when minimizing.

    It does NOT mask. `MaskedComposite` already restricts the gradient to the
    surround, because the composite depends on `x` only through `(1 - m) * x`, so
    the chain rule delivers `(1 - m) * dL/dcomposite` before this op ever sees it.
    Multiplying by `(1 - m)` a second time would apply the factor TWICE,
    attenuating the gradient to `(1 - m)^2` across the soft transition band.

    Their deposited code composes exactly this
    (`jiakunf/Fu_et_al_2026`, `analysis/base.py`, `generate_exc_surround`):

        gradient_f = Compose([GaussianBlur(blur_sigma)])                  # max
        gradient_f = Compose([GaussianBlur(blur_sigma), MultiplyBy(-1)])  # min

    Their `ops.py` does carry a `MaskGradient` (line 617), but it is a bare
    `x * (1 - mask)` with no blur, and the deposited driver never calls it -- the
    mask reaches the gradient through the transform, as here.

    sigma = 0 disables the blur, used only by the analytic test.
    """

    def __init__(self, sigma: float = 0.0, negate: bool = False):
        self.blur = fv_ops.GaussianBlur(sigma) if sigma and sigma > 0 else None
        self.negate = negate

    @fv_utils.varargin
    def __call__(self, grad, **kwargs):
        if self.blur is not None:
            grad = self.blur(grad)
        return -grad if self.negate else grad


class ChangeSurroundStd:
    """post_update: surround mean 0, RMS contrast `surround_std`.

    Reimplementation of the featurevis op their code imports but the public package
    does not ship (`jiakunf/Fu_et_al_2026`, `featurevis/ops.py`).

    The RMS is computed over the outside-mask pixels only (DEVIATIONS.md D1): their
    `ChangeSurroundStd(mei, mask, [center_std, surround_std])` takes a two-element
    list keyed to two regions, and their source weights the standard deviation by
    `1 - mask`.

    Non-differentiable by design, matching their note that the contrast
    renormalization step "was not differentiable".
    """

    def __init__(self, mask: torch.Tensor, surround_std: float):
        self.mask, self.surround_std = mask, surround_std
        self._outside = 1.0 - mask
        self._n_outside = float(self._outside.sum())

    @fv_utils.varargin
    def __call__(self, x, **kwargs):
        # weighted mean and std over the surround region only
        mean = (self._outside * x).sum() / self._n_outside
        centered = self._outside * (x - mean)
        std = torch.sqrt((centered ** 2).sum() / self._n_outside)
        return centered * (self.surround_std / (std + 1e-12))


def compose(*operations):
    """featurevis Compose, ignoring Nones for convenience."""
    return fv_utils.Compose([op for op in operations if op is not None])


# ---- the three steps ----------------------------------------------------

def optimize_mei(model, cfg: Config, seed: int | None = None,
                 return_trace: bool = False) -> torch.Tensor:
    """Step 1: find the most exciting input, full field.

    `return_trace=True` also returns the `fevals` list, which must be checked for
    a plateau before any optimization result on a nonlinear model is believed.
    Drivers must call this rather than re-implementing the `gradient_ascent` call,
    so that a change to the surround gradient cannot miss a code path.
    """
    seed = cfg.seed if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    x0 = torch.randn(1, 1, cfg.img_px, cfg.img_px) * cfg.contrast_center

    grad_f = fv_ops.GaussianBlur(cfg.grad_blur_sigma) if cfg.grad_blur_sigma > 0 else None
    post = compose(
        ZeroMean(),
        fv_ops.ChangeStd(cfg.contrast_center),
        ClipRange(*cfg.pixel_clip_range),
    )

    mei, fevals, _ = gradient_ascent(
        model, x0,
        gradient_f=grad_f,
        post_update=post,
        optim_name="SGD",
        step_size=cfg.opt_step_size,
        num_iterations=cfg.opt_mei_iters,
        print_iters=cfg.opt_mei_iters + 1,
    )
    return (mei.detach(), fevals) if return_trace else mei.detach()


def mask_for(mei: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Step 2: threshold-and-hull mask around the MEI, optionally dilated."""
    m = create_mask(
        mei.numpy().squeeze(),
        zscore_thresh=cfg.mask_zscore_thresh,
        closing_iters=cfg.mask_closing_iters,
        blur_sigma=cfg.mask_blur_sigma,
        dilation_px=cfg.mask_dilation_px,
    )
    return torch.tensor(m, dtype=torch.float32)[None, None]


def optimize_surround(model, mei: torch.Tensor, mask: torch.Tensor, cfg: Config,
                      objective: str = "max", seed: int | None = None,
                      return_trace: bool = False) -> torch.Tensor:
    """Step 3: hold the MEI fixed inside the mask, optimize outside.

    `objective` is "max" (facilitatory) or "min" (suppressive). Run BOTH, always --
    the suppressive surround is a mechanism-specific fingerprint and the strongest
    available check that a model is implemented correctly, because unlike
    facilitation it has a known expected form.

    Returns the full composite (MEI + optimized surround).
    """
    if objective not in ("max", "min"):
        raise ValueError(f"objective must be 'max' or 'min', got {objective!r}")

    seed = cfg.seed if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    x0 = torch.randn(1, 1, cfg.img_px, cfg.img_px) * cfg.surround_init_scale
    composite = MaskedComposite(mei, mask)

    out, fevals, _ = gradient_ascent(
        model, x0,
        transform=composite,
        gradient_f=SurroundGradient(cfg.grad_blur_sigma, negate=(objective == "min")),
        post_update=compose(
            ChangeSurroundStd(mask, cfg.contrast_surround),
            ClipRange(*cfg.pixel_clip_range),
        ),
        optim_name="SGD",
        step_size=cfg.opt_step_size,
        num_iterations=cfg.opt_surround_iters,
        print_iters=cfg.opt_surround_iters + 1,
    )
    comp = composite(out.detach())
    return (comp, fevals) if return_trace else comp


def center_only(mei: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The MEI-alone stimulus: MEI masked to the center, no surround."""
    return mask * mei
