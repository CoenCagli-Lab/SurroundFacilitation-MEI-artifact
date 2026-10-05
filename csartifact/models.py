"""The three models.

    Model A   LN Gabor. One linear filter, ELU+1 output. No normalization pool, no
              inhibition, NO SURROUND MECHANISM OF ANY KIND. Any facilitatory
              surround it yields is produced entirely by the center/surround
              definition.
    Model B   Heeger divisive normalization, untuned global pool. d_0 = 2 primary;
              d_0 = 0 is the faithful replication of their published model.
    Model C   Model B's pool plus a multiplicative facilitatory gain drawn from an
              ORTHOGONALLY tuned annular pool. The positive control.

Every model takes an image tensor of shape (B, 1, H, W) in z-scored input space and
returns a scalar response. All are differentiable with respect to the input and run
on CPU. `featurevis.gradient_ascent` calls the model as a plain callable.

ONE d_0 PER MODEL, shared by the suppressive and facilitatory pools.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F


def gabor(
    size: int,
    sigma_px: float,
    sf_cpp: float,
    theta: float = 0.0,
    phase: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
    aspect: float = 1.0,
) -> np.ndarray:
    """Mean-subtracted Gabor. `center` is offset from image center, in pixels."""
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2.0
    xx = xx - center[0]
    yy = yy - center[1]
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    envelope = np.exp(-(xr ** 2 + (aspect * yr) ** 2) / (2.0 * sigma_px ** 2))
    g = envelope * np.cos(2.0 * np.pi * sf_cpp * xr + phase)
    return g - g.mean()


class Model(ABC):
    """Base class. Subclasses implement `response`."""

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def response(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar response of the target neuron."""

    def drive(self, x: torch.Tensor) -> torch.Tensor:
        """Linear drive of the target neuron, before any nonlinearity."""
        raise NotImplementedError

    @property
    @abstractmethod
    def center_filter(self) -> np.ndarray:
        """The target neuron's linear filter, as a 2-D array.

        Ground truth for mask-versus-filter comparisons. Every model here has a
        well-defined center filter; that is what makes the leakage argument
        checkable.
        """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.response(x)


# ---- Model A ------------------------------------------------------------

class LNGabor(Model):
    """Model A -- LN Gabor.

    Analytic ground truth:

      * the MEI under an RMS contrast constraint c is the matched filter,
            x* = c * N * f / ||f||
      * maximum drive from a region R is c * N * ||f . 1_R|| when the RMS is taken
        over the whole field
      * the MEI's RMS is whole-field but the surround's is surround-only, so
        maximum fractional facilitation is
            (c_surround / c_center) * ||f . (1-m)|| / ||f . m|| * sqrt(N_s) / N
        with N_s = sum(1-m).
    """

    def __init__(self, cfg, theta: float = 0.0, phase: float = 0.0,
                 center: tuple[float, float] = (0.0, 0.0)):
        super().__init__(cfg)
        self._filter = gabor(
            size=cfg.img_px,
            sigma_px=cfg.sigma_px,
            sf_cpp=cfg.sf_cpp,
            theta=theta,
            phase=phase,
            center=center,
            aspect=cfg.gabor_aspect,
        )
        self._f = torch.tensor(self._filter, dtype=torch.float32)[None, None]

    @property
    def center_filter(self) -> np.ndarray:
        return self._filter

    def drive(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self._f).sum()

    def response(self, x: torch.Tensor) -> torch.Tensor:
        # ELU+1 follows their choice, made to keep gradients alive during
        # optimization. Soft rectifier, value 1 at drive 0.
        return F.elu(self.drive(x)) + 1.0

    # ---- analytic reference -------------------------------------------
    def analytic_mei(self, contrast: float) -> np.ndarray:
        """The exact MEI: the filter scaled to full-field RMS `contrast`."""
        f = self._filter - self._filter.mean()
        return f * (contrast / f.std())


# ---- Model B ------------------------------------------------------------

# Building 10,000 Gabors costs a second or two and ~350 MB; models constructed with
# identical pool parameters share one bank.
_BANK_CACHE: dict = {}


def gabor_bank(size: int, sigma_px: float, sf_cpp: float, aspect: float,
               n: int, seed: int, chunk: int = 512):
    """`n` Gabors with random orientation, position and phase.

    Returns `(bank, params)` where `bank` is (n, size*size) float32, one
    mean-subtracted filter per row, and `params` records the sampled values so
    Model C can reuse them.

    Spatial frequency and envelope sigma are FIXED across the pool, per their
    Methods. Built in chunks: the full (n, size, size) intermediates would run to
    several GB at n = 10,000.
    """
    key = (int(size), round(float(sigma_px), 9), round(float(sf_cpp), 12),
           float(aspect), int(n), int(seed))
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]

    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, np.pi, n)              # period pi -- orientation, not direction
    phase = rng.uniform(0.0, 2.0 * np.pi, n)
    half = (size - 1) / 2.0
    cx = rng.uniform(-half, half, n)                # uniform over the image, as "global" implies
    cy = rng.uniform(-half, half, n)

    rows, cols = np.mgrid[:size, :size]
    yy = (rows - half).astype(np.float32)
    xx = (cols - half).astype(np.float32)

    bank = np.empty((n, size * size), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        t = theta[s:e, None, None].astype(np.float32)
        p = phase[s:e, None, None].astype(np.float32)
        X = xx[None] - cx[s:e, None, None].astype(np.float32)
        Y = yy[None] - cy[s:e, None, None].astype(np.float32)
        ct, st = np.cos(t), np.sin(t)
        xr = X * ct + Y * st
        yr = -X * st + Y * ct
        env = np.exp(-(xr ** 2 + (aspect * yr) ** 2) / (2.0 * np.float32(sigma_px) ** 2))
        g = env * np.cos(2.0 * np.pi * np.float32(sf_cpp) * xr + p)
        g -= g.mean(axis=(1, 2), keepdims=True)     # every filter mean-subtracted, as Model A
        bank[s:e] = g.reshape(e - s, -1)

    params = {"theta": theta, "phase": phase, "center": np.stack([cx, cy], axis=1),
              "sf": np.full(n, float(sf_cpp))}
    _BANK_CACHE[key] = (bank, params)
    return bank, params


class HeegerDN(Model):
    """Model B -- untuned global divisive normalization around a fixed target Gabor.

        y_j  = <f_j, x>           j = 1..pool_size, random orientation/position/phase
        p_j  = ReLU(y_j - d_0)                  <- pool uses ReLU
        ybar = mean_j(p_j)
        R_i  = (ELU(y_i) + 1) / (1 + ybar)      <- target uses ELU+1

    The asymmetry is theirs and is preserved: the pool is rectified with ReLU, the
    target neuron uses ELU+1.

    The pool is global -- every filter sits somewhere in the 93x93 image -- so
    suppression comes from the normalizer and is unaffected by where the
    center/surround mask is drawn.

    CORRECTNESS CHECK: the suppressive surround must come out texture-like, NOT
    antiphase-collinear like Model A. If it is antiphase-collinear the normalization
    pool is not engaged and the model is wrong. Catch this before believing any
    facilitation result.

    On spatial frequency: the pool holds a single SF (2.5 cyc/deg), identical to the
    target's, so every unit is iso-SF with the target and the pool is precisely
    SF-tuned by construction. Scale-broadbandness is not a requirement on this
    model -- a single-SF pool cannot produce it, and SF coverage is not relevant to
    the proof of principle. The signature that matters is broadband in ORIENTATION
    and spatially distributed.

    The target filter defaults to the same canonical Gabor as Model A -- theta 0,
    phase 0, image center -- so that facilitation in A and B is measured on the same
    center and the dilation curves are comparable. The pool does not include the
    target; the difference is O(1/pool_size).
    """

    def __init__(self, cfg, theta: float = 0.0, phase: float = 0.0,
                 center: tuple[float, float] = (0.0, 0.0)):
        super().__init__(cfg)
        self._filter = gabor(
            size=cfg.img_px,
            sigma_px=cfg.sigma_px,
            sf_cpp=cfg.sf_cpp,
            theta=theta,
            phase=phase,
            center=center,
            aspect=cfg.gabor_aspect,
        )
        self._f = torch.tensor(self._filter, dtype=torch.float32)[None, None]
        self.theta, self.phase = theta, phase

        bank, params = gabor_bank(
            size=cfg.img_px,
            sigma_px=cfg.sigma_px,
            sf_cpp=cfg.sf_cpp,
            aspect=cfg.gabor_aspect,
            n=cfg.pool_size,
            seed=cfg.pool_seed,
        )
        self._pool = torch.from_numpy(bank)         # (n, P), shared across instances
        self.pool_params = params

    @property
    def center_filter(self) -> np.ndarray:
        return self._filter

    def drive(self, x: torch.Tensor) -> torch.Tensor:
        """Linear drive of the target, before any nonlinearity."""
        return (x * self._f).sum()

    def pool_drive(self, x: torch.Tensor) -> torch.Tensor:
        """`y_j = <f_j, x>` for the whole pool. THE dominant cost: 10,000 x 8,649.

        Model C needs this twice per response -- once for `pool_mean` and again for
        `gain`. Rather than recompute it, `response()` evaluates it ONCE and threads
        the result through via the optional `pool_drive=` argument on the methods
        below. That halves Model C from 56.8 to 28.5 ms per optimizer iteration.

        Explicit threading rather than a cache: memoizing on tensor identity breaks
        autograd, because when the same `x` object is used for two separate backward
        passes the second receives a tensor whose graph has already been freed.
        Passing the value down has no such hazard.
        """
        return self._pool @ x.reshape(-1)

    def pool_responses(self, x: torch.Tensor, pool_drive=None) -> torch.Tensor:
        """Rectified response of every pool unit. (n,)

        `d_0` raises the rectification threshold, so the normalization pool needs
        more drive before it engages.
        d_0 = 0 is the unthresholded model, which is Model B as published.
        Note ReLU(ReLU(y) - d_0) == ReLU(y - d_0) for d_0 >= 0, so subtracting
        before the rectifier is the same operation, done once.
        """
        y = (self.pool_drive(x) if pool_drive is None else pool_drive) \
            - self.cfg.opt_pool_threshold
        return F.relu(y)

    def pool_mean(self, x: torch.Tensor, pool_drive=None) -> torch.Tensor:
        """`ybar`, the normalization signal. Exposed for diagnostics."""
        return self.pool_responses(x, pool_drive).mean()

    def response(self, x: torch.Tensor) -> torch.Tensor:
        pd = self.pool_drive(x)
        return (F.elu(self.drive(x)) + 1.0) / (1.0 + self.pool_mean(x, pd))


# ---- Model C ------------------------------------------------------------

def facilitatory_weights(pool_params: dict, sigma_px: float, kappa_ori: float,
                         peak_radius_sigma: float, width_sigma: float,
                         theta: float = 0.0,
                         center: tuple[float, float] = (0.0, 0.0),
                         ori_offset_deg: float = 90.0) -> np.ndarray:
    """Orientation-tuned, annular weighting over the pool. Unnormalized weights.

    The spatial term is a RING at `peak_radius_sigma` rather than a disc centered on
    the target. That is what puts the facilitatory mechanism outside the classical
    receptive field.

    `ori_offset_deg` is the pool's preferred orientation RELATIVE to the center
    filter. The primary is **90 degrees, orthogonal**, and the reason is that the
    control has to be legible: leakage into the center filter's own fringe
    reproduces the CENTER's orientation, so an iso-oriented facilitatory mechanism
    produces structure that looks exactly like the artifact it exists to be
    distinguished from. At 90 degrees the optimized facilitatory surround is
    orthogonal to the center wherever the mechanism drives it and iso-oriented
    wherever leakage does, so the two are separable by eye in a single image.
    """
    theta_pref = theta + np.deg2rad(ori_offset_deg)
    v_ori = np.exp(kappa_ori * (np.cos(2.0 * (pool_params["theta"] - theta_pref)) - 1.0))

    d = np.hypot(pool_params["center"][:, 0] - center[0],
                 pool_params["center"][:, 1] - center[1])
    r_peak = peak_radius_sigma * sigma_px
    width = width_sigma * sigma_px
    v_ann = np.exp(-((d - r_peak) ** 2) / (2.0 * width ** 2))

    v = v_ori * v_ann
    total = v.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"degenerate facilitatory weights (sum={total}); check "
                         f"kappa_ori_surr, h_peak_radius, h_width")
    return v


class FacilitatoryDN(HeegerDN):
    """Model C -- divisive normalization with a GENUINE facilitatory surround.

    The positive control.

        p_j   = ReLU(y_j - d_0)                        shared with the suppressive pool
        ybar  = mean_j(p_j)                            suppression, as Model B
        s_i   = sum_j(v_j * p_j) / sum_j(v_j)          facilitation, annular ORTHOGONAL
        R_i   = ( ELU( y_i * (1 + beta * s_i) ) + 1 ) / (1 + ybar)

    with the facilitatory weighting

        v_ori_j = exp(kappa_ori_surr * (cos(2*(theta_j - theta_pref)) - 1))
                  where theta_pref = theta_i + 90 deg          ORTHOGONAL
        v_ann_j = exp(-(d_ij - h_peak_radius)^2 / (2 * h_width^2))   ANNULAR, not a disc
        v_j     = v_ori_j * v_ann_j

    BOTH surround mechanisms are weighted sums over the SAME 10,000-unit bank, built
    the same way as Model B, and both read the same rectified responses `p_j`.

    MULTIPLICATIVE ONLY. The gain multiplies `y_i` before the nonlinearity, so
    `y_i = 0` gives `R_i = 1/(1 + ybar)` whatever `s_i` is: the surround cannot drive
    the cell on its own, it only modulates drive the center supplies. An additive
    surround would make `f_i` plus the facilitatory field one larger linear filter
    with no recoverable center.

    DISPLACED. `v_ann` is an annulus peaking at `h_peak_radius` (3.5) envelope sigma,
    so the facilitatory pool draws on units whose receptive fields sit OUTSIDE the
    classical RF. 
    """

    def __init__(self, cfg, theta: float = 0.0, phase: float = 0.0,
                 center: tuple[float, float] = (0.0, 0.0)):
        super().__init__(cfg, theta=theta, phase=phase, center=center)
        v = facilitatory_weights(
            self.pool_params,
            sigma_px=cfg.sigma_px,
            kappa_ori=cfg.kappa_ori_surr,
            peak_radius_sigma=cfg.h_peak_radius,
            width_sigma=cfg.h_width,
            theta=theta,
            center=center,
            ori_offset_deg=cfg.h_ori_offset_deg,
        )
        self.facilitatory_pool_weights = v
        self._v = torch.tensor(v / v.sum(), dtype=torch.float32)

        d = np.hypot(self.pool_params["center"][:, 0] - center[0],
                     self.pool_params["center"][:, 1] - center[1])
        self._pool_distance_px = d

    @property
    def effective_facilitatory_pool_size(self) -> float:
        """Kish effective sample size of the facilitatory weighting.

        2066 of 10,000 at the primary settings.
        """
        v = self.facilitatory_pool_weights
        return float(v.sum() ** 2 / (v ** 2).sum())

    @property
    def weight_outside_fraction(self) -> float:
        """Fraction of facilitatory weight on units whose RF centers lie beyond 2
        envelope sigma -- i.e. outside the classical RF the mask approximates."""
        v = self.facilitatory_pool_weights
        return float(v[self._pool_distance_px > 2.0 * self.cfg.sigma_px].sum() / v.sum())

    def weight_inside_mask_fraction(self, mask) -> float:
        """Share of facilitatory pool weight on units whose RF centers fall INSIDE
        the MEI mask.

        Must stay small, and does: 1.57% at beta = 0.3, measured on the orthogonal
        primary. The concern this quantity exists to test is whether a larger beta
        grows the mask until the measured "center" starts absorbing the mechanism.

        Takes the mask rather than reading it at construction: the pool weighting
        itself must never depend on the mask.
        """
        msk = mask.detach().numpy().squeeze() if hasattr(mask, "detach") else np.asarray(mask)
        half = (self.cfg.img_px - 1) / 2.0
        col = np.clip(np.rint(self.pool_params["center"][:, 0] + half).astype(int),
                      0, self.cfg.img_px - 1)
        row = np.clip(np.rint(self.pool_params["center"][:, 1] + half).astype(int),
                      0, self.cfg.img_px - 1)
        inside = msk[row, col] > 0.5
        v = self.facilitatory_pool_weights
        return float(v[inside].sum() / v.sum())

    @property
    def weight_peak_radius_sigma(self) -> float:
        """Weight-averaged RF-center distance, in envelope sigmas.

        The REALIZED annulus centroid: 3.77 sigma against a nominal `h_peak_radius`
        of 3.5, because unit density in the bank rises with radius.
        """
        v = self.facilitatory_pool_weights
        return float((v * self._pool_distance_px).sum() / v.sum() / self.cfg.sigma_px)

    def surround_drive(self, x: torch.Tensor, pool_drive=None) -> torch.Tensor:
        """`s_i`, the facilitatory pool's weighted mean response.

        Reads the same `p_j = ReLU(y_j - d_0)` as the suppressive pool, so one
        threshold governs both surround mechanisms. Non-negative by construction.
        """
        return (self._v * self.pool_responses(x, pool_drive)).sum()

    def gain(self, x: torch.Tensor, pool_drive=None) -> torch.Tensor:
        """`1 + beta * s_i`. Multiplies the center's drive.

        No rectifier of its own: `s_i` is a weighted mean of non-negative `p_j`.
        """
        return 1.0 + self.cfg.beta * self.surround_drive(x, pool_drive)

    def response(self, x: torch.Tensor) -> torch.Tensor:
        pd = self.pool_drive(x)          # computed ONCE, used by gain and pool_mean
        return (F.elu(self.drive(x) * self.gain(x, pd)) + 1.0) \
            / (1.0 + self.pool_mean(x, pd))


MODELS = {"model_a": LNGabor, "model_b": HeegerDN, "model_c": FacilitatoryDN}


def build(cfg):
    """Instantiate the model `cfg` names."""
    return MODELS[cfg.model](cfg)
