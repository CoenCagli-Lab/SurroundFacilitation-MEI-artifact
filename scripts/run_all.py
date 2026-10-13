"""Regenerate everything from scratch: payloads, numbers, figures.

This script is for someone who wants to check that the
committed payloads really do come out of this code.

To simply look at the results, read the committed `results/*.json`
and redraw the figures in seconds with `--only figures`

    python3 scripts/run_all.py                  everything
    python3 scripts/run_all.py --only figures   redraw only, seconds
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# (stage, script)
STAGES = [
    ("annulus", "sweep_annulus.py"),
    ("stimuli", "make_stimuli.py"),
    ("dilation", "sweep_dilation.py"),
    ("numbers", "emit_numbers.py"),
    ("figures", "fig2_dilation.py"),
    ("figures", "fig3_annulus.py"),
    ("figures", "fig4_stimuli.py"),
    ("figures", "suppfig2_model_a.py"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", metavar="STAGE",
                    choices=sorted({s for s, _ in STAGES}),
                    help="run only these stages")
    args = ap.parse_args()

    selected = [(s, f) for s, f in STAGES
                if args.only is None or s in args.only]

    print(f"running {len(selected)} steps\n", flush=True)

    t0 = time.perf_counter()
    for stage, script in selected:
        t = time.perf_counter()
        print(f"=== {stage}: {script} ===", flush=True)
        result = subprocess.run([sys.executable, str(HERE / script)], cwd=REPO)
        if result.returncode != 0:
            print(f"\n{script} failed with exit code {result.returncode}; stopping. "
                  f"Nothing after this point has run.", file=sys.stderr)
            return result.returncode
        print(f"    [{(time.perf_counter() - t) / 60:.1f} min]\n", flush=True)

    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")
    print("Results are in results/, figures in figures/, numbers in NUMBERS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
