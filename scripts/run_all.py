"""Regenerate everything from scratch: payloads, numbers, figures.

FOR VERIFICATION RATHER THAN INSPECTION. A reader who wants to look at the results
should read the committed `results/*.json` and redraw the figures in seconds with
`--only figures`; this script exists for someone who wants to check that the
committed payloads really do come out of this code.

A little over an hour, dominated by the dilation sweep, which re-optimizes the
surround at every step. Measured on the machine that produced the committed payloads:

    annulus     6 min
    stimuli     7 min
    dilation   53 min      prints each series against the committed values as it
                           finishes, so a drifting run shows up along the way
    numbers     1 min
    figures    <1 min

Cheapest first, so that a run which is not going to reproduce the committed numbers
is caught early rather than after an hour.

On the machine that produced them the sweeps are deterministic under their fixed
seeds, and a successful run reproduces every number in the committed payloads
exactly; only the `written` timestamps differ. On DIFFERENT hardware, BLAS builds
or torch/SciPy versions expect last-digit differences instead: these are float32
optimizations accumulated over thousands of iterations. That is not a failure.

`--only figures` redraws from whatever is already in `results/` and optimizes
nothing.

THE WIDGET IS DELIBERATELY ABSENT, and so is the dense sweep that feeds it. Neither
`sweep_dilation_dense.py` nor `make_widget.py` is a stage here.
Rebuild it on its own when it changes:

    python3 scripts/sweep_dilation_dense.py
    python3 scripts/make_widget.py

Both are analytic and take seconds, so there is no cost to leaving them out.

    python3 scripts/run_all.py                  everything
    python3 scripts/run_all.py --only figures   redraw only, seconds
    python3 scripts/run_all.py --list           show the stages and exit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# (stage, script, rough minutes)
STAGES = [
    ("annulus", "sweep_annulus.py", 6),
    ("stimuli", "make_stimuli.py", 7),
    ("dilation", "sweep_dilation.py", 53),
    ("numbers", "emit_numbers.py", 1),
    ("figures", "fig2_dilation.py", 0),
    ("figures", "fig3_annulus.py", 0),
    ("figures", "fig4_stimuli.py", 0),
    ("figures", "suppfig2_model_a.py", 0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", metavar="STAGE",
                    choices=sorted({s for s, _, _ in STAGES}),
                    help="run only these stages")
    ap.add_argument("--list", action="store_true", help="show the stages and exit")
    args = ap.parse_args()

    selected = [(s, f, m) for s, f, m in STAGES
                if args.only is None or s in args.only]

    if args.list:
        for stage, script, minutes in selected:
            print(f"  {stage:<9} {script:<22} {minutes or '<1':>3} min")
        print(f"  {'':<9} {'':<22} {sum(m for _, _, m in selected):>3} min total")
        return 0

    total = sum(m for _, _, m in selected)
    print(f"running {len(selected)} steps, roughly {total} min\n", flush=True)

    t0 = time.perf_counter()
    for stage, script, _ in selected:
        t = time.perf_counter()
        print(f"=== {stage}: {script} ===", flush=True)
        result = subprocess.run([sys.executable, str(HERE / script)], cwd=REPO)
        if result.returncode != 0:
            print(f"\n{script} failed with exit code {result.returncode}; stopping. "
                  f"Nothing after this point has run.", file=sys.stderr)
            return result.returncode
        print(f"    [{(time.perf_counter() - t) / 60:.1f} min]\n", flush=True)

    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")
    print("The sweeps are deterministic, so `git diff results/` should be empty "
          "except for the timestamps in each meta block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
