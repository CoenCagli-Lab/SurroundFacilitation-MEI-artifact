"""Experiment configuration.

Every value below produced a number printed in one of the two manuscripts
-- `NUMBERS.md` indexes them all, with each value's status --
so none of it is adjustable without re-running.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Published optimization settings, per model. (step_size, mei_iters, surround_iters)
#
# Model A takes their macaque CNN row; Models B and C take their
# divisive-normalization row, whose surround count is 3000, not 1000.  DEVIATIONS.md
# D4 records them: the settings differ per model, and the difference matters.
MODEL_OPTIM: dict[str, tuple[float, int, int]] = {
    "model_a": (10.0, 1000, 1000),
    "model_b": (0.1, 1000, 3000),
    "model_c": (0.1, 1000, 3000),
}

# Primary pool threshold d_0. ONE threshold governs every surround mechanism --
# suppressive and facilitatory alike -- because both are weighted sums over the
# same rectified pool responses p_j = ReLU(y_j - d_0).
#
# The surround has LOWER SENSITIVITY than the center, so the thresholded
# models are primary for B and C. The one exception is the faithful replication of
# their published Heeger model, which rectifies at zero: that is Model B with
# `pool_threshold=0.0`, the "as published" configuration, not the primary.
PRIMARY_THRESHOLDS: dict[str, float] = {
    "model_a": 0.0,      # no pool
    "model_b": 2.0,
    "model_c": 2.0,      # one threshold, both surround mechanisms
}


@dataclass(frozen=True)
class Config:
    # ---- geometry -------------------------------------------------------
    img_px: int = 93
    deg_per_image: float = 2.67          # the value stated in their Methods
    gabor_sigma_deg: float = 0.2
    gabor_sf_cpd: float = 2.5
    gabor_aspect: float = 1.0

    # ---- model ----------------------------------------------------------
    model: Literal["model_a", "model_b", "model_c"] = "model_a"
    pool_size: int = 10_000              # Models B and C
    # Pool identity is part of the MODEL, so it must NOT ride on `seed`: anything
    # that varies `seed` to re-run an optimization from a different init would
    # otherwise be comparing different models.
    pool_seed: int = 0
    # d_0 in p_j = ReLU(y_j - d_0). None means "this model's primary value".
    # Read the resolved number off `opt_pool_threshold`, never off this field.
    # Set 0.0 explicitly for Model B as published.
    pool_threshold: float | None = None

    # ---- Model C facilitatory pool --------------------------------------
    # Peaks OUTSIDE the center filter envelope, so the facilitatory pool draws on
    # units whose receptive fields sit beyond the classical RF. If it drew on units
    # overlapping f_i, gradient ascent would spread the MEI over both and the mask
    # would grow to encompass the whole mechanism.
    # `beta` is the facilitatory strength. At smaller values the optimizer could not
    # tell the facilitatory surround apart from the center (Supplement, Section 3.1).
    beta: float = 0.3
    kappa_ori_surr: float = 2.0          # orientation tuning of the facilitatory pool
    h_peak_radius: float = 3.5           # units of target envelope sigma
    h_width: float = 1.0                 # a Gaussian SD in radius, not a FWHM
    # Preferred orientation of the facilitatory pool, RELATIVE to the center filter.
    # 90 deg = ORTHOGONAL. Leakage into the center filter's fringe
    # reproduces the center's own orientation, so an iso-oriented facilitatory
    # mechanism (offset 0) is visually indistinguishable from the artifact.
    # At 90 the two separate by eye.
    h_ori_offset_deg: float = 90.0

    # ---- optimization ---------------------------------------------------
    grad_blur_sigma: float = 1.0         # the value stated in their Methods (mouse). Fixed.
    # None means "use this model's published value from MODEL_OPTIM"; read the
    # resolved numbers off the `opt_*` properties, never off these fields.
    step_size: float | None = None
    num_iterations: int | None = None            # MEI
    num_iterations_surround: int | None = None   # surround
    contrast_center: float = 0.05
    contrast_surround: float = 0.10      # the ratio is what matters
    # The surround starts from noise at this scale, as in their `create_surround`. The
    # first update rescales it to the surround contrast, so it only affects which local
    # optimum is reached, not the contrast constraint.
    surround_init_scale: float = 0.5
    pixel_clip_range: tuple[float, float] = (-1.7876, 2.1919)

    # ---- mask -----------------------------------------------------------
    mask_zscore_thresh: float = 1.5
    mask_closing_iters: int = 2
    mask_blur_sigma: float = 1.0
    mask_dilation_px: int = 0

    # ---- run ------------------------------------------------------------
    seed: int = 0

    # ---- derived --------------------------------------------------------
    @property
    def ppd(self) -> float:
        return self.img_px / self.deg_per_image

    @property
    def sigma_px(self) -> float:
        return self.gabor_sigma_deg * self.ppd

    @property
    def sf_cpp(self) -> float:
        return self.gabor_sf_cpd / self.ppd

    # ---- resolved optimization settings ---------------------------------
    # Always read these, not the raw fields.

    @property
    def opt_step_size(self) -> float:
        return MODEL_OPTIM[self.model][0] if self.step_size is None else self.step_size

    @property
    def opt_mei_iters(self) -> int:
        return MODEL_OPTIM[self.model][1] if self.num_iterations is None else self.num_iterations

    @property
    def opt_surround_iters(self) -> int:
        if self.num_iterations_surround is None:
            return MODEL_OPTIM[self.model][2]
        return self.num_iterations_surround

    @property
    def opt_pool_threshold(self) -> float:
        if self.pool_threshold is None:
            return PRIMARY_THRESHOLDS[self.model]
        return self.pool_threshold

    def __post_init__(self):
        if self.model not in MODEL_OPTIM:
            raise ValueError(f"no published optimization settings for model {self.model!r}")
