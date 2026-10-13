"""Regenerate everything from scratch: payloads, numbers, figures.

This script is for someone who wants to check that the
committed payloads really do come out of this code.

To simply look at the results, read the committed `results/*.json`
and redraw the figures in seconds with `--only figures`

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
    print("Results are in results/, figures in figures/, numbers in NUMBERS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
