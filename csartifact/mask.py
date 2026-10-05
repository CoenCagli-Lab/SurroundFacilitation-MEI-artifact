"""MEI mask construction.

Reimplemented from the algorithm description in their published code, and copied from
neither deposit that carries it. Both do, and the two are the same sequence:
`compute_mei_mask` in jiakunf/Fu_et_al_2026 and `create_mask_from_mei` in
lucabaroni/center-surround. Neither repository carries a license.

The algorithm:

    1. z-score the MEI
    2. threshold on ABSOLUTE value: |z| > 1.5
    3. binary closing, 2 iterations
    4. keep the largest connected component
    5. convex hull of that component
    6. Gaussian blur, sigma 1 px, rescale to [0, 1]

Dilation is applied after the hull and before the blur. That ordering is what the
dilation sweep measured; it is not interchangeable with blurring first.

Two mask conventions appear in the printed numbers and they do not agree, so every
printed number names the one it used -- the `mask` tag in the conventions column of
`NUMBERS.md`:

    soft      the blurred mask in [0, 1] as `create_mask` returns it. THEIR mask:
              their procedure ends at the blur and returns this. It is also what
              the optimizer sees.
    binary    the hard-thresholded mask, m > 0.5. OURS, and derived from the
              soft one: a blurred mask has a transition band and so does not
              partition energy, which the closed form needs it to do.

`mask_radius_px` hard-thresholds in both cases -- an equivalent-circle radius is
only defined for a set -- so it is the *energy*, *norm* and *headroom* quantities
that the convention selects, not the area.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure, morphology


def create_mask(
    mei: np.ndarray,
    zscore_thresh: float = 1.5,
    closing_iters: int = 2,
    blur_sigma: float = 1.0,
    dilation_px: int = 0,
) -> np.ndarray:
    """Return a soft mask in [0, 1] with the same shape as `mei`."""
    mei = np.asarray(mei, dtype=np.float64).squeeze()
    if mei.ndim != 2:
        raise ValueError(f"expected a 2-D MEI, got shape {mei.shape}")

    z = (mei - mei.mean()) / (mei.std() + 1e-12)
    binary = np.abs(z) > zscore_thresh

    if closing_iters:
        binary = ndimage.binary_closing(binary, iterations=closing_iters)

    binary = _largest_component(binary)
    binary = morphology.convex_hull_image(binary)

    # Dilation 0 is the mask Fu et al.'s procedure produces. Positive values grow it
    # outward; NEGATIVE values erode it inward. 
    # Erosion exists for the widget (`scripts/make_widget.py`) and feeds
    # no printed figure; no printed number and no assertion uses a negative value.
    #
    # Either way the morphology is applied to the BINARY region, before the blur, so
    # the transition band is produced once by the blur and is the same width at every
    # setting. 
    if dilation_px > 0:
        binary = ndimage.binary_dilation(binary, iterations=dilation_px)
    elif dilation_px < 0:
        binary = ndimage.binary_erosion(binary, iterations=-dilation_px)
        if not binary.any():
            raise ValueError(
                f"erosion by {-dilation_px} px removes the mask entirely; the "
                f"undilated mask is about 27 px across, so it cannot survive an "
                f"erosion approaching its radius")

    mask = ndimage.gaussian_filter(binary.astype(np.float64), blur_sigma)
    peak = mask.max()
    return mask / peak if peak > 0 else mask


def _largest_component(binary: np.ndarray) -> np.ndarray:
    labels = measure.label(binary)
    if labels.max() == 0:
        return binary
    counts = np.bincount(labels.ravel())
    counts[0] = 0                      # ignore background
    return labels == counts.argmax()


def mask_radius_px(mask: np.ndarray, level: float = 0.5) -> float:
    """Equivalent-circle radius of the mask, in pixels."""
    area = float((np.asarray(mask) > level).sum())
    return float(np.sqrt(area / np.pi))



def filter_split(filt: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """L2 norm of the filter inside and outside the mask.

    For a linear unit under an RMS contrast constraint, the maximum drive
    obtainable from a region is proportional to the filter's L2 norm restricted
    to that region (matched filter). So these two numbers give the analytic
    leakage bound directly.

    The printed energy fractions (`geom.*` in `NUMBERS.md`) are ratios of
    energies, ||f.m||^2 / ||f||^2, which partition exactly under a binary mask;
    the norms are their square roots.
    """
    filt = np.asarray(filt).squeeze()
    mask = np.asarray(mask).squeeze()
    inside = float(np.sqrt(((filt * mask) ** 2).sum()))
    outside = float(np.sqrt(((filt * (1.0 - mask)) ** 2).sum()))
    return inside, outside


def analytic_headroom(filt: np.ndarray, mask: np.ndarray,
                      contrast_center: float, contrast_surround: float) -> float:
    """Maximum fractional gain in linear drive from an optimal surround.

        (c_surround / c_center) * ||f . (1 - m)|| / ||f . m|| * sqrt(N_s / N)

    `N` is the full field area
    `N_s = sum(1 - m)` is the *effective* surround area

    The last factor is there because the surround's RMS contrast is taken over the
    surround pixels only (DEVIATIONS.md D1), while the MEI's is taken over the full
    field. Contrast enters only through the RATIO `c_s / c_c`.

    This is an upper bound. With a binary mask around the ideal MEI it is the
    closed-form 32% the Supplement prints (`artifact.drive_ratio_closedform`). With
    the soft mask around the numerical MEI, the configuration Figure 2 uses, it is
    28.71% (the widget's starting value, `drive_ratio_pct` at dilation 0 in
    `results/model_a_dilation_dense.json`). Undilated, the optimizer reaches 95% of
    that bound -- 27.31% against 28.71% -- and much less once the mask is dilated.
    The widget plots this bound.
    """
    inside, outside = filter_split(filt, mask)
    bound = (contrast_surround / contrast_center) * outside / inside
    n_sur = float(np.asarray(1.0 - mask).sum())
    return bound * np.sqrt(n_sur / float(np.asarray(mask).size))
