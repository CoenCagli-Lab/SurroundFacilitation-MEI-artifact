# DEVIATIONS

**Section A — deviations from Fu et al.'s method.** Where this implementation departs
from theirs, or we had to choose because their Methods or deposited code is silent.

**Section B — implementation notes.** Choices and corrections belonging to this
repository rather than to their method. Named descriptively.

Every value quoted below is the current measured one, from `results/numbers.json`
or, where noted, from the result files in `results/`.

---

# A. Deviations from Fu et al.'s method

## D1 — Surround contrast normalization region

**Resolved by reading their source.** `ChangeSurroundStd` is absent from public
`cajal/featurevis`, which the surround algorithm is credited to. It is present in the
featurevis copy vendored inside their mouse deposit — `jiakunf/Fu_et_al_2026`,
`featurevis/ops.py`, `class ChangeSurroundStd` — which sets `sur_mask = 1 - mask` and
takes `sur_std` as a standard deviation of the optimization variable weighted by it.

So the RMS is taken **over the surround region only**, not over the whole field.

## D2 — Pixel clipping

Their `create_mei` and `create_surround` both apply `ClipRange(-1.7876, 2.1919)` in
z-scored input space, every iteration. The paper mentions only that the contrast
constraint was chosen to keep pixels inside the display range.

**Implemented at their values, and always applied.** It matters in principle because
clipping is a hard constraint on how much energy the optimizer can pack into a thin
annulus adjacent to the mask — which is the mechanism this repository claims produces
the facilitation. Applying it means the demonstration runs under their constraint
rather than a more permissive one.

## D3 — Licensing

**The repository is MIT licensed**, © 2026 Ruben Coen-Cagli.
`LICENSE` carries the grant and `NOTICE`
records the one modification to the vendored dependency.

The audit below is retained as the **basis for that decision**.
In summary: `third_party/featurevis` is public `cajal/featurevis` under
MIT, its LICENSE is retained unmodified in that directory, and MIT is the natural
choice for this repository because it is compatible with that single obligation.
Nothing here is copied from any of their three deposited repositories, none of which
carries a license. The mask implementation is written from the algorithm description
their paper states, rather than from their file.

The mask construction reimplements a published algorithm, and reading their source served
only to confirm that the 1.5 SD threshold applies to `|z|`.

## D4 — Optimization hyperparameters are per model

Their Methods specify different settings per model:
When we run Models B and C at the macaque CNN's 1000 steps and step size of 10,
Model B finds no facilitatory surround at all, returning suppression for both objectives.
With 3000 steps and smaller step size of 0.1, it finds facilitation,
so the step size was two orders of magnitude too large
for the normalization model B.

| their model | step | MEI steps | surround steps |
|---|---|---|---|
| macaque CNN — our Model A | 10 | 1000 | 1000 |
| divisive normalization — our Models B and C | 0.1 | 1000 | **3000** |

Anything that constructs a config by copying another model's
must re-resolve these, which is why they are read off
properties rather than stored fields.

## D5 — Model B pool construction

Their Methods give the pool as "10,000 LN Gabor simple cells, orientation, position
and phase randomly sampled", with spatial frequency 2.5, σ 0.2, aspect ratio 1. Four
things are left open, and the choices are:

| open point | choice |
|---|---|
| sampling distributions | orientation U[0, π), phase U[0, 2π), center uniform over the 93 × 93 field |
| pool filter amplitude | not normalized, matching Model A's `gabor()` convention |
| is the target in its own pool? | no — the difference is O(1/pool_size), about 1e-4 |
| target filter identity | the canonical Model A Gabor, θ 0, phase 0, centered |

The pool is single spatial frequency, so it is iso-SF with the target
by construction and the suppressive surround is correspondingly tuned to that one scale.

## D6 — Annulus stimuli: hard edge, no renormalization after blanking

**Contrast is NOT restored after blanking.** Removing the center lowers the image's
RMS. Rescaling the annulus back would answer a different question — what an annulus at
full contrast *could* do — and would inflate the surviving response by construction,
since the surround was optimized at 0.10 RMS. The reported residual is therefore a
lower bound.

**The edge is hard**, `keep = (dist >= r)`. No raised-cosine alternative is
computed.

## Provenance audit, for D3

Three of their repositories carry **no license**, and absent a license the default is
all rights reserved. **Re-checked live against the GitHub API on 2026-09-21:**

| repository | license | root LICENSE / COPYING / NOTICE | last pushed |
|---|---|---|---|
| `jiakunf/Fu_et_al_2026` — mouse analysis | **none** | none; no README either | 2026-03-20 |
| `lucabaroni/center-surround` — macaque analysis | **none** | none | 2026-03-25 |
| `sinzlab/probabilistic-center-surround` | **none** | none | 2026-04-01 |
| `cajal/featurevis` | **MIT** | `LICENSE`, © 2019 Cajal MICrONS Team | 2022-05-03 |

All four are public. For the three unlicensed ones the API's `license` field is null,
there is no license-like file at the repository root, and neither
`lucabaroni/center-surround`'s `pyproject.toml` nor
`sinzlab/probabilistic-center-surround`'s `setup.py` declares one.

The vendored `cajal/featurevis` LICENSE here is byte-identical to the live one.

**Nothing in this repository is copied from any of them.** Two files cite them, and
both citations record *reading*, not copying:

| file | what it says |
|---|---|
| `csartifact/mask.py` | states the mask is reimplemented from the algorithm description and **not** copied from either deposit that carries it |
| `csartifact/optimize.py` | cites their `analysis/base.py` for a two-line factual observation — that their surround `gradient_f` composes a Gaussian blur, and a negation when minimizing, and nothing else |

`sinzlab/probabilistic-center-surround` was searched once, for an implementation of
their Supplementary Figure 5, and nothing was found or taken. No reference to it
survives anywhere in this repository.

**What the mask implementation derives from.** `create_mask` is written from a
seven-step description — z-score, `|z| > 1.5`, binary closing ×2,
largest connected component, convex hull, Gaussian blur σ 1, rescale to [0, 1] —
which was obtained by reading their own implementation of it. Both deposits carry
that implementation and the two are the same sequence statement for statement, down
to the comments: `compute_mei_mask` in `jiakunf/Fu_et_al_2026`, `analysis/base.py`
— alongside `generate_surround_mask`, the same steps applied to the optimized
surround — and `create_mask_from_mei` in `lucabaroni/center-surround`,
`surroundmodulation/utils/mask.py`. Every step is a single
stock call into `scipy.ndimage` or `skimage`, so the code is independent even though
the sequence is theirs. That sequence is also described in their paper, and reading
their source served only to confirm that the 1.5 SD threshold applies to `|z|`.

**`ChangeSurroundStd` is an independent reimplementation.** Theirs computes a
weighted standard deviation with `statsmodels.DescrStatsW` and returns the assembled
composite. Ours uses torch reductions, normalizes the surround alone, and returns
only the surround term — the composite is assembled separately by `MaskedComposite`.
A different decomposition, not a transcription.

**Nothing beyond `third_party/featurevis` carries a license obligation.** That copy is
public `cajal/featurevis`, MIT, © 2019 Cajal MICrONS Team, and the LICENSE file is
present and unmodified. It is demonstrably the public repository rather than the copy
vendored inside their unlicensed mouse deposit: that copy carries seven operations
absent from public featurevis — `ChangeSurroundStd`, `ChangeCenterStd`,
`PostSurroundStd`, `ChangeStdJoint`, `MaskGradient`, `MaskImage` and `ChangeMean` —
and **none of the seven is present here**, which is precisely why `ChangeSurroundStd`
had to be reimplemented at all. The one modification is the documented
`scipy.signal.windows.gaussian` patch. Everything else the repository depends on —
numpy, scipy, scikit-image, torch, matplotlib, pillow — is an ordinary permissively
licensed dependency and is not vendored.

**Both loose ends are now closed.** The repository carries `LICENSE` — MIT, © 2026
Ruben Coen-Cagli — and `NOTICE`, which records the one modification to the vendored
copy. `third_party/featurevis/LICENSE` is retained unmodified and governs that
directory.

---

# B. Implementation notes

Choices belonging to this repository rather than to their method.

## Model C's β

β sets Model C's facilitatory strength. Model C is the positive control here, not
their model. **β = 0.3.** At smaller values the optimizer could not tell the
facilitatory surround apart from the center (Supplement, Section 3.4.1).

## Model C's facilitatory pool sits outside the measured center

A facilitatory gain could in principle inflate the MEI, and so the mask built from
it, until the measured center absorbed the mechanism it is supposed to be separate
from. Model C's facilitation would then have the same explanation this work gives for
the original result: a larger measured center, not a surround mechanism.

It does not happen. Model C's mask (1.74 σ) is the same size as Model B's, and only
**1.57%** of the facilitatory pool's weight falls inside it
(`weight_inside_mask_fraction` at 0 px in `results/model_c_dilation.json`). **So Model C's facilitation cannot be attributed to a
larger measured center.**

## Model C's facilitatory annulus is orthogonal to the center

`h_ori_offset_deg = 90` is the primary; 0 would implement an iso-oriented surround.

**An iso-oriented control cannot be told apart from the artifact by eye.** Leakage
past the mask boundary reproduces the *center* filter's orientation, because that is
the structure the center filter reads. An iso-oriented facilitatory mechanism
produces the same orientation for a different reason, so the two can be told apart
only quantitatively, by the mask-dilation sweep (Figure 2): a real facilitatory
mechanism keeps its facilitation once the mask reaches about 3 σ, and leakage does
not.

At 90° one image separates the artifact from the genuine facilitatory surround:
mechanism-driven structure is orthogonal, leaked structure stays iso-oriented,
and Figure 4 shows exactly that — Model C's facilitatory
surround carries clear horizontal structure flanking the vertical center.

## The payloads are reproducible — identical numbers here, within tolerance elsewhere

**On this machine, identical numbers.** Every full regeneration of the payloads has
returned **identical numbers**; only the `written` timestamps change. Under the fixed
seeds
(`seed = 0` for the MEI and the surround initialization, `pool_seed = 0`
for the pool itself) the sweeps are deterministic, so a re-run here reproduces the
committed numbers exactly rather than closely. 
