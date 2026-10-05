"""Build `docs/mask-radius.html` — the interactive widget.

A VISUAL DEMO TO BUILD INTUITION. One slider, Model A, the
analytic bound precomputed at each position. Nothing is optimized in the browser and
nothing is optimized here either: the curve values come from
`results/model_a_dilation_dense.json`; the mask at each slider position is built with
`mask_for` from the MEI in the same `stimuli.npz` the figures draw; and the image
behind it is the model's filter, `model.center_filter`.

This script computes one thing the payload does not carry: the mask's outline at each
slider position, as an SVG polygon. Outlines are arrays, and the payload contract
keeps arrays out of the JSON, so they are inlined straight into the page instead.

The page plots the ANALYTIC BOUND on response facilitation (bound x d_c/(d_c+1)), not
the optimizer result: at Fu et al.'s mask it reads +28.0%, where the Supplement prints
the optimizer's 26.6% (in drive, the bound is 28.71% against the printed 27.3%). Only
response is plotted; the payload's drive values feed the console summary.

NOT PART OF `run_all.py`, by decision. That script regenerates what the manuscripts
print; this page feeds no printed figure and no printed number, so rebuilding it on
every verification run would only add a ~70 KB file to the diff. Run it directly when
the page or the dense sweep changes -- it takes about a second.

    python3 scripts/make_widget.py
"""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage import measure

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/featurevis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from csartifact.config import Config
from csartifact.models import build
from csartifact.optimize import mask_for
# The two envelopes are drawn in Supplementary Figure 2 and here, so their colors
# have one definition. The curve colors below are this page's own: they have no
# counterpart in the figures, where Figure 2's series are colored per model.
from _style import ENVELOPE_2SIGMA, ENVELOPE_3SIGMA

RESULTS = REPO / "results"
DOCS = REPO / "docs"

# ONE link, it points to the repository.
REPO_URL = "https://github.com/CoenCagli-Lab/SurroundFacilitation-MEI-artifact"


def filter_png(img: np.ndarray) -> str:
    """The filter as a base64 PNG, on a linear grayscale.
    """
    v = float(np.abs(img).max())
    grey = np.clip((img / v) * 0.5 + 0.5, 0, 1)
    png = Image.fromarray((grey * 255).astype(np.uint8), mode="L").resize(
        (372, 372), Image.NEAREST)
    buf = io.BytesIO()
    png.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def outline(mask: np.ndarray) -> list[list[float]]:
    """The mask's 0.5 contour as SVG polygon points, in image pixel coordinates."""
    contours = measure.find_contours(mask, 0.5)
    biggest = max(contours, key=len)
    step = max(1, len(biggest) // 120)                 # ~120 points is plenty
    return [[round(float(c), 2), round(float(r), 2)] for r, c in biggest[::step]]


def main() -> int:
    dense = json.loads((RESULTS / "model_a_dilation_dense.json").read_text())
    cfg = Config(model="model_a")
    model = build(cfg)

    with np.load(RESULTS / "stimuli.npz") as z:
        mei = torch.tensor(z["model_a_mei"], dtype=torch.float32)[None, None]

    steps = []
    for i, px in enumerate(dense["dilation_px"]):
        mask = mask_for(mei, dataclasses.replace(cfg, mask_dilation_px=px))
        steps.append({
            "px": px,
            "r_sigma": round(dense["mask_radius_sigma"][i], 4),
            "diam_deg": round(2 * dense["mask_radius_sigma"][i] * cfg.gabor_sigma_deg, 4),
            "drive": round(dense["drive_ratio_pct"][i], 4),
            "resp": round(dense["response_pct"][i], 4),
            "poly": outline(mask.numpy().squeeze()),
        })

    data = {
        "steps": steps,
        "default": dense["dilation_px"].index(0),
        "fu_sigma": round(dense["mask_radius_sigma"][dense["dilation_px"].index(0)], 4),
        "env2_px": round(2 * cfg.sigma_px, 3),
        "env3_px": round(3 * cfg.sigma_px, 3),
        "img": filter_png(model.center_filter),
    }

    # The control's range and starting position are DERIVED, not typed in. The page
    # must open at Fu et al.'s own mask -- that is the comparison everything else is
    # read against.
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    html = html.replace("__MAX__", str(len(steps) - 1))
    html = html.replace("__DEFAULT__", str(data["default"]))
    html = html.replace("__REPO__", REPO_URL)
    html = html.replace("__ENV2__", ENVELOPE_2SIGMA)
    html = html.replace("__ENV3__", ENVELOPE_3SIGMA)
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "mask-radius.html"
    out.write_text(html)

    at0 = data["default"]
    print(f"  {len(steps)} slider positions, {steps[0]['px']} to {steps[-1]['px']} px")
    print(f"  default at index {at0}: mask {steps[at0]['r_sigma']}sigma, "
          f"drive {steps[at0]['drive']}%, response {steps[at0]['resp']}%")
    print(f"  wrote docs/{out.name}  ({out.stat().st_size / 1024:.0f} KB, self-contained)")
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The boundary decides the surround effect</title>
<style>
  :root{
    --ink:#16191d; --muted:#5d6470; --faint:#9aa2ae; --rule:#dfe3e9;
    --bg:#fbfbfc; --card:#ffffff;
    --fac:#0173B2; --sup:#DE8F05; --env:__ENV3__; --env2:__ENV2__; --mask:#111417;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
       -webkit-font-smoothing:antialiased}
  .wrap{max-width:1120px;margin:0 auto;padding:34px 24px 40px}
  .sub{color:var(--muted);margin:0 0 24px;max-width:78ch}
  .sub a{color:var(--fac)}
  .panels{display:grid;grid-template-columns:minmax(280px,.85fr) minmax(340px,1.15fr);
          gap:26px;align-items:start}
  .card{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:18px}
  .card h2{font-size:12px;letter-spacing:.07em;text-transform:uppercase;
           color:var(--faint);margin:0 0 12px;font-weight:600}
  svg{display:block;width:100%;height:auto}
  .legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:14px;font-size:13.5px;
          color:var(--muted)}
  .legend i{display:inline-block;width:24px;height:0;border-top-width:3px;
            margin-right:7px;vertical-align:middle}
  .note{margin:12px 0 0;font-size:12px;line-height:1.5;color:var(--faint)}
  .note b{color:var(--muted);font-weight:600}
  .control{margin:26px 0 0;background:var(--card);border:1px solid var(--rule);
           border-radius:10px;padding:18px 20px 16px}
  .ctl-top{display:flex;justify-content:space-between;align-items:baseline;
           gap:16px;flex-wrap:wrap;margin-bottom:12px}
  .ctl-top strong{font-size:13px;letter-spacing:.02em}
  .vals{font-variant-numeric:tabular-nums;font-size:13px;color:var(--muted)}
  .vals b{font-size:19px;letter-spacing:-.01em}
  .vals .f{color:var(--fac)} .vals .s{color:var(--sup)}
  .radius{font-variant-numeric:tabular-nums;color:var(--faint);font-size:12.5px;
          display:block;margin-top:2px}
  input[type=range]{width:100%;margin:0;accent-color:var(--fac);height:26px}
  /* Each label sits under the slider position it names, with a tick mark pointing
     at it. The positions are computed from the data in the script below. */
  .ticks{position:relative;height:36px;color:var(--faint);font-size:11px;
         line-height:1.25;font-variant-numeric:tabular-nums;margin-top:2px}
  .ticks span{position:absolute;top:8px;white-space:nowrap;text-align:center;
              transform:translateX(-50%)}
  .ticks span::before{content:"";position:absolute;left:50%;top:-8px;width:1px;
                      height:6px;background:currentColor}
  .ticks span.l{transform:none;text-align:left}
  .ticks span.l::before{left:0}
  .ticks span.r{transform:translateX(-100%);text-align:right}
  .ticks span.r::before{left:auto;right:0}
  @media (max-width:520px){.ticks span.l{display:none}.ticks small{display:none}}
  @media (max-width:780px){.panels{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">

<p class="sub"><strong>Model A is a linear Gabor filter followed by rectification.
It does not have any surround mechanism.</strong> The apparent surround facilitation
and suppression are the artifact discussed in the Letter.
Drag the slider to control the size of the mask.
&nbsp;·&nbsp; <a href="__REPO__">Code and data</a></p>

<div class="panels">

  <div class="card">
    <h2>The filter, and the boundary you drew</h2>
    <svg viewBox="-3 -3 99 99" role="img" aria-label="Gabor filter with mask boundary">
      <image id="filt" x="0" y="0" width="93" height="93" preserveAspectRatio="none"/>
      <circle id="e3" cx="46" cy="46" fill="none" stroke="var(--env)"
              stroke-width=".7" stroke-dasharray="3 1.6"/>
      <circle id="e2" cx="46" cy="46" fill="none" stroke="var(--env2)"
              stroke-width=".9" stroke-dasharray=".8 1.1"/>
      <polygon id="mask" fill="none" stroke="var(--mask)" stroke-width="1.15"/>
    </svg>
    <div class="legend">
      <span><i style="border-top:3px solid var(--mask)"></i>mask you set</span>
      <span><i style="border-top:3px dotted var(--env2)"></i>2&sigma;</span>
      <span><i style="border-top:3px dashed var(--env)"></i>filter envelope (3&sigma;)</span>
    </div>
  </div>

  <div class="card">
    <svg viewBox="0 0 348 300" role="img"
         aria-label="Apparent surround facilitation and suppression against mask radius">
      <g id="grid"></g>
      <!-- the slider's position, drawn FIRST so the fixed reference stays legible
           when the two coincide -- which they do at the default -->
      <line id="vline" y1="18" y2="256" stroke="#dfe3e9" stroke-width="5"/>
      <line id="fuline" y1="18" y2="256" stroke="var(--mask)" stroke-width="1.1"
            stroke-dasharray="5 3"/>
      <text id="fulabel" y="13" font-size="11" fill="var(--mask)"
            text-anchor="middle">Fu et al. 2026</text>
      <path id="curve_f" fill="none" stroke="var(--fac)" stroke-width="2.6"/>
      <path id="curve_s" fill="none" stroke="var(--sup)" stroke-width="2.6"/>
      <circle id="marker_f" r="5.5" fill="var(--fac)" stroke="#fff" stroke-width="2"/>
      <circle id="marker_s" r="5.5" fill="var(--sup)" stroke="#fff" stroke-width="2"/>
      <text x="196" y="40" font-size="12" fill="var(--fac)">facilitatory surround</text>
      <text x="196" y="240" font-size="12" fill="var(--sup)">suppressive surround</text>
      <text x="195" y="290" font-size="12" fill="#9aa2ae" text-anchor="middle">
        mask radius, in filter envelope &sigma;</text>
      <text font-size="12" fill="#9aa2ae" transform="rotate(-90 14 137)"
            text-anchor="middle">
        <tspan x="14" y="132">apparent surround facilitation (%)</tspan>
        <tspan x="14" y="146">analytic</tspan>
      </text>
    </svg>
  </div>
</div>

<div class="control">
  <div class="ctl-top">
    <div>
      <strong>Mask radius</strong>
      <span class="radius">mask <span id="rs">1.93</span>&sigma; &nbsp;·&nbsp;
        diameter <span id="rd">0.77</span>&deg; &nbsp;·&nbsp;
        <span id="rp">0</span> px</span>
    </div>
    <div class="vals">
      facilitation <b class="f" id="vf">+28.0%</b> &nbsp;·&nbsp;
      suppression <b class="s" id="vs">&minus;28.0%</b>
    </div>
  </div>
  <input type="range" id="sl" min="0" max="__MAX__" step="1" value="__DEFAULT__">
  <div class="ticks" id="ticks"></div>
</div>

</div>

<script>
const D = __DATA__;
const S = D.steps;
const $ = id => document.getElementById(id);

$("filt").setAttribute("href", D.img);
$("e2").setAttribute("r", D.env2_px);
$("e3").setAttribute("r", D.env3_px);

// ---- the curve panel ----
// Symmetric about zero: Model A's suppressive surround is its facilitatory one in
// antiphase, so the two curves are exact mirrors and the plot says so by construction
// rather than by assertion.
const PL = 50, PR = 336, PT = 18, PB = 256, MID = (PT + PB) / 2;
const xs = S.map(s => s.r_sigma);
const X0 = xs[0], X1 = xs[xs.length - 1];
const YMAX = 80;                                   // the eroded end reaches 72
const px = v => PL + (v - X0) / (X1 - X0) * (PR - PL);
const py = v => MID - (v / YMAX) * (MID - PT);

let g = "";
for (const t of [80, 40, 0, -40, -80]) {
  g += `<line x1="${PL}" y1="${py(t).toFixed(1)}" x2="${PR}" y2="${py(t).toFixed(1)}"
         stroke="${t === 0 ? "#c9ced6" : "#eceff3"}" stroke-width="1"/>`
     + `<text x="${PL - 7}" y="${(py(t) + 3.5).toFixed(1)}" font-size="11" fill="#9aa2ae"
         text-anchor="end">${t > 0 ? "+" + t : t}</text>`;
}
for (const t of [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]) {
  if (t < X0 || t > X1) continue;
  g += `<text x="${px(t).toFixed(1)}" y="${PB + 16}" font-size="11" fill="#9aa2ae"
         text-anchor="middle">${t.toFixed(1)}</text>`;
}
$("grid").innerHTML = g;

const path = sign => S.map((s, i) =>
  (i ? "L" : "M") + px(s.r_sigma).toFixed(2) + " " + py(sign * s.resp).toFixed(2)
).join(" ");
$("curve_f").setAttribute("d", path(1));
$("curve_s").setAttribute("d", path(-1));

// Fixed reference at the mask Fu et al.'s procedure actually produces. It stays put
// while the slider moves, so the reader can always see how far they have dragged
// from the published boundary -- which is the comparison the page exists to make.
const xfu = px(D.fu_sigma);
$("fuline").setAttribute("x1", xfu); $("fuline").setAttribute("x2", xfu);
$("fulabel").setAttribute("x", xfu);

const fmt = (v, sign) => (sign > 0 ? "+" : "\u2212") + v.toFixed(1) + "%";

function draw(i) {
  const s = S[i];
  const pts = s.poly.map(p => p.join(",")).join(" ");
  $("mask").setAttribute("points", pts);

  const xv = px(s.r_sigma);
  $("vline").setAttribute("x1", xv); $("vline").setAttribute("x2", xv);
  $("marker_f").setAttribute("cx", xv); $("marker_f").setAttribute("cy", py(s.resp));
  $("marker_s").setAttribute("cx", xv); $("marker_s").setAttribute("cy", py(-s.resp));

  $("vf").textContent = fmt(s.resp, 1);
  $("vs").textContent = fmt(s.resp, -1);
  $("rs").textContent = s.r_sigma.toFixed(2);
  $("rd").textContent = s.diam_deg.toFixed(2);
  $("rp").textContent = s.px;
}

// Deterministic start whatever the browser restored. Browsers repopulate form
// controls on reload, so the markup's value alone is not enough -- the page must
// open at Fu et al.'s own mask every time, because every other position is read as
// a departure from it.
// Slider labels, each placed where the slider's thumb sits for the value it names.
// The thumb's CENTRE travels from THUMB/2 to (width - THUMB/2), not across the full
// width, so a fraction f of the range sits at THUMB/2 + f * (width - THUMB).
const THUMB = 16;
const N = S.length - 1;
const atIndex = i => `calc(${THUMB / 2}px + ${i / N} * (100% - ${THUMB}px))`;
// fractional slider index at which the mask radius reaches `sigma`
function indexAtSigma(sigma) {
  for (let i = 0; i < N; i++) {
    if (xs[i] <= sigma && sigma <= xs[i + 1]) {
      return i + (sigma - xs[i]) / (xs[i + 1] - xs[i]);
    }
  }
  return null;
}
const sig = v => v.toFixed(1) + "\u03c3";
const TICKS = [
  {i: 0, cls: "l", html: "eroded"},
  {i: D.default, html: sig(S[D.default].r_sigma) + "<br><small>Fu et al. 2026</small>"},
  {i: indexAtSigma(3.0), html: sig(3.0)},
  {i: N, cls: "r", html: sig(S[N].r_sigma) + "<br><small>envelope enclosed</small>"},
];
$("ticks").innerHTML = TICKS.filter(t => t.i !== null).map(t =>
  `<span class="${t.cls || ""}" style="left:${atIndex(t.i)}">${t.html}</span>`
).join("");

$("sl").value = D.default;
draw(D.default);
$("sl").addEventListener("input", e => draw(+e.target.value));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
