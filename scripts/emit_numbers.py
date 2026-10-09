"""Emit `results/numbers.json` and `NUMBERS.md`.

Every number the two manuscripts print gets a key, emitted by code, carrying the
conventions it was measured under. Nothing here is typed in from a document: the
scalars are computed live or read from the sweep payloads, so this index cannot
drift out of step with the code.

Each entry records the conventions its value was measured under -- which mask and
which MEI -- because several quantities are printed under more than one and the
numbers differ. Every number assumes the 93-pixel image spans 2.67 deg, the value
stated in their Methods.

The live part is Model A, plus the MEIs of Models B and C for their mask radii.
About a minute.

    python3 scripts/emit_numbers.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/featurevis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from csartifact.config import Config
from csartifact.mask import (analytic_headroom, create_mask,
                             filter_split, mask_radius_px)
from csartifact.measure import facilitation_measures
from csartifact.models import build
from csartifact.optimize import (mask_for, optimize_mei,
                                 optimize_surround)
from _payload import RESULTS

torch.set_num_threads(4)

SERIES = ["model_a", "model_b_d0_2", "model_b_d0_0", "model_c"]
numbers: dict[str, dict] = {}


# One spelling per model, normalized in put() rather than at the call sites, so no
# new key can introduce a variant. The registry is meant to be filtered by this tag:
# when the same model was written "A" in some entries and "model_a" in others, a
# filter on one silently missed the other.
MODEL_TAG = {
    "model_a": "A",                 "A": "A",
    "model_b": "B_d0_2",            "B": "B_d0_2",          "B_d0_2": "B_d0_2",
    "model_b_d0_2": "B_d0_2",
    "model_b_d0_0": "B_d0_0",       "B_d0_0": "B_d0_0",
    "model_c": "C",                 "C": "C",
}


def put(key, value, **tags):
    if tags.get("model") is not None:
        tag = tags["model"]
        if tag not in MODEL_TAG:
            raise KeyError(f"{key}: unknown model tag {tag!r}; add it to MODEL_TAG")
        tags["model"] = MODEL_TAG[tag]
    numbers[key] = {"value": value,
                    "tags": {k: v for k, v in tags.items() if v is not None}}


def read(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


t0 = time.perf_counter()

# ---- live: Model A -------------------------------------------------------

cfg = Config(model="model_a")
model = build(cfg)
f = model.center_filter

ideal = model.analytic_mei(cfg.contrast_center)
soft_ideal = create_mask(ideal, cfg.mask_zscore_thresh, cfg.mask_closing_iters,
                         cfg.mask_blur_sigma)
binary = (soft_ideal > 0.5).astype(float)

mei = optimize_mei(model, cfg)
mask = mask_for(mei, cfg)
m = mask.numpy().squeeze()
comp_max = optimize_surround(model, mei, mask, cfg, objective="max")
comp_min = optimize_surround(model, mei, mask, cfg, objective="min")
print(f"  Model A done ({time.perf_counter() - t0:.0f} s)", flush=True)

# ---- geometry ------------------------------------------------------------

r_bin = mask_radius_px(binary)
r_soft = mask_radius_px(m)
energy_in_bin = float(((f * binary) ** 2).sum() / (f ** 2).sum())
norm = float(np.sqrt((f ** 2).sum()))
in_bin, out_bin = filter_split(f, binary)

G = dict(model="A", mask="binary", mei="ideal")
put("geom.mask_diameter_px", round(2 * r_bin), **G)
put("geom.env_2sigma_diameter_px", round(4 * cfg.sigma_px),
    model="A", mei="ideal")
put("geom.mask_area_px_closedform", int((binary > 0.5).sum()), **G)
put("geom.image_area_px", int(binary.size))
put("geom.surround_area_px", int(binary.size - (binary > 0.5).sum()), **G)
put("geom.energy_inside_mask", 100 * energy_in_bin, **G)
put("geom.energy_outside_mask", 100 * (1 - energy_in_bin), **G)
put("geom.norm_inside_mask", 100 * in_bin / norm, **G)
put("geom.norm_outside_mask", 100 * out_bin / norm, **G)
put("geom.mask_radius_sigma.A", r_soft / cfg.sigma_px,
    model="A", mask="soft", mei="numerical")
put("geom.energy_inside_mask_soft",
    100 * float(((f * m) ** 2).sum() / (f ** 2).sum()),
    model="A", mask="soft", mei="numerical")

# ---- the other models' masks and MEI fidelity ----------------------------

put("mei.corr_with_filter.A",
    float(np.corrcoef(mei.numpy().ravel(), f.ravel())[0, 1]),
    model="A", mei="numerical")

OTHERS = [
    ("geom.mask_radius_sigma.B", "mei.corr_with_filter.B", Config(model="model_b"), "B"),
    ("geom.mask_radius_sigma.C", "mei.corr_with_filter.C", Config(model="model_c"), "C"),
]
for radius_key, corr_key, other_cfg, label in OTHERS:
    other = build(other_cfg)
    other_mei = optimize_mei(other, other_cfg)
    other_mask = mask_for(other_mei, other_cfg)
    put(radius_key,
        mask_radius_px(other_mask.numpy().squeeze()) / other_cfg.sigma_px,
        model=label, mask="soft", mei="numerical")
    put(corr_key, float(np.corrcoef(other_mei.numpy().ravel(),
                                    other.center_filter.ravel())[0, 1]),
        model=label, mei="numerical")
    print(f"  Model {label} mask done ({time.perf_counter() - t0:.0f} s)", flush=True)

# ---- Model A core result -------------------------------------------------

bound_ideal = analytic_headroom(f, binary, cfg.contrast_center, cfg.contrast_surround)
d_mei_ideal = cfg.contrast_center * cfg.img_px * np.sqrt(((f * binary) ** 2).sum())
put("artifact.drive_ratio_closedform", 100 * bound_ideal,
    **G, measure="drive")
put("artifact.response_facilitation_closedform",
    100 * bound_ideal * float(d_mei_ideal / (d_mei_ideal + 1)),
    **G, measure="response")

fm = facilitation_measures(model, mei, mask, comp_max)
fn = facilitation_measures(model, mei, mask, comp_min)
T = dict(model="A", mask="soft", mei="numerical")
put("artifact.drive_ratio", 100 * fm["drive_ratio"],
    **T, measure="drive", objective="maximizing")
put("artifact.response_facilitation", 100 * fm["response_ratio"], **T,
    measure="response", objective="maximizing")
put("artifact.response_suppression", 100 * fn["response_ratio"], **T,
    measure="response", objective="minimizing")

# ---- the Letter's headline: a rule, not a value -------------------------
fac = numbers["artifact.response_facilitation"]["value"]
sup = numbers["artifact.response_suppression"]["value"]
put("artifact.letter_headline",
    {"rule": "min(|facilitation|, |suppression|) > 25",
     "facilitation": fac, "suppression": sup,
     "margin": min(abs(fac), abs(sup)) - 25.0,
     "holds": bool(min(abs(fac), abs(sup)) > 25.0)}, **T)

# ---- dilation ------------------------------------------------------------

dil = {s: read(f"{s}_dilation") for s in SERIES}

# Model B with d0 = 0: its undilated mask radius, the upper end of the printed
# "1.74-1.78 sigma for B and C". Read from the sweep, which builds the mask the
# same way as the radii above.
put("geom.mask_radius_sigma.B_d0_0", dil["model_b_d0_0"]["mask_radius_sigma"][0],
    model="B_d0_0", mask="soft", mei="numerical")
at9 = {s: d["dilation_px"].index(9) for s, d in dil.items()}
put("dilation.expansion_px_at_3sigma", 9, model="A")
for s, short in (("model_a", "A"), ("model_b_d0_2", "B"), ("model_c", "C")):
    put(f"dilation.facilitation_at_3sigma.{short}",
        dil[s]["facilitatory"]["response_pct"][at9[s]],
        model=short, measure="response", objective="maximizing")
    put(f"dilation.suppression_undilated.{short}",
        dil[s]["suppressive"]["response_pct"][0],
        model=short, measure="response", objective="minimizing")
    put(f"dilation.suppression_at_25px.{short}",
        dil[s]["suppressive"]["response_pct"][-1],
        model=short, measure="response", objective="minimizing")

put("dilation.dilation_px", dil["model_a"]["dilation_px"])
for s in SERIES:
    put(f"dilation.response_pct.{s}", dil[s]["facilitatory"]["response_pct"],
        model=s, measure="response", objective="maximizing")
    put(f"dilation.drive_pct.{s}", dil[s]["facilitatory"]["drive_pct"],
        model=s, measure="drive", objective="maximizing")

# ---- annulus -------------------------------------------------------------

ann = {s: read(f"{s}_annulus") for s in SERIES}
put("annulus.radii_px", ann["model_a"]["radii_px"], model="A")
for s in SERIES:
    put(f"annulus.response_rel_blank.{s}", ann[s]["response_rel_blank"],
        model=s, measure="response", stimulus="annulus")
    put(f"annulus.drive_pct_of_full.{s}", ann[s]["drive_pct_of_full"],
        model=s, measure="drive", stimulus="annulus")

# ---- write ---------------------------------------------------------------

RESULTS.mkdir(exist_ok=True)
(RESULTS / "numbers.json").write_text(json.dumps({
    "meta": {
        "generated_by": "scripts/emit_numbers.py",
        "scope": "every number printed in the Letter or its Supplement, "
                 "one key per value",
        "field_deg": cfg.deg_per_image,
    },
    "numbers": dict(sorted(numbers.items())),
}, indent=1) + "\n")

print(f"\n  wrote results/numbers.json -- {len(numbers)} keys "
      f"in {(time.perf_counter() - t0) / 60:.1f} min")


# ---- NUMBERS.md ----------------------------------------------------------
# A reader's index of results/numbers.json: one row per printed number, with the
# conventions it was measured under.

SECTIONS = [
    ("Geometry", "geom."),
    ("MEI fidelity", "mei."),
    ("Model A core result and headline", "artifact."),
    ("Dilation (Figure S1g)", "dilation."),
    ("Annulus (Figure S1h)", "annulus."),
]


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        head = ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in v[:4])
        return f"array[{len(v)}] — {head}{', …' if len(v) > 4 else ''}"
    if isinstance(v, dict):
        return "; ".join(f"{k} {fmt(x)}" for k, x in v.items())
    return str(v)


lines = [
    "# NUMBERS",
    "",
    "Every number printed across the two manuscripts, emitted by",
    "`scripts/emit_numbers.py` into `results/numbers.json` and indexed here. Every",
    "key carries the conventions its value was measured under: which mask (`binary`",
    "or `soft`) and which MEI (`ideal` or `numerical`).",
    "",
    f"Every number assumes the 93-pixel image spans {cfg.deg_per_image} deg.",
    "",
    "Every value here is an output of the code in this repository.",
    "",
]
written = 0
for title, prefixes in SECTIONS:
    prefixes = (prefixes,) if isinstance(prefixes, str) else prefixes
    rows = [(k, v) for k, v in sorted(numbers.items()) if k.startswith(prefixes)]
    if not rows:
        continue
    written += 1
    lines += [f"## {title}", "", "| key | value | conventions |", "|---|---|---|"]
    for k, v in rows:
        tags = ", ".join(f"{a} {b}" for a, b in v["tags"].items()) or "—"
        value = fmt(v["value"]).replace("|", "\\|")    # a bare | would end the table cell
        lines.append(f"| `{k}` | {value} | {tags} |")
    lines.append("")

lines += [
    "---",
    "",
    "To rebuild this index from the payloads already in `results/`:",
    "`python3 scripts/emit_numbers.py`, about a minute. To recompute the values",
    "themselves first: `python3 scripts/run_all.py`, a little over an hour.",
]
(REPO / "NUMBERS.md").write_text("\n".join(lines) + "\n")
print(f"  wrote NUMBERS.md -- {len(numbers)} rows across {written} sections")
