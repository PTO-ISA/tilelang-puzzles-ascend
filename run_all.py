"""Run the whole ladder and print an honest status table.

    python run_all.py                      # everything
    python run_all.py --tier torch         # one tier (torch | asc | pto)
    python run_all.py --kernel per_token   # one kernel, all tiers
    python run_all.py --role puzzle        # check the unsolved puzzles instead
    python run_all.py --tier torch --jobs 8

Each variant runs in its own process, because the NPU tiers re-launch themselves
under the CPU simulator. The torch tier is CPU-only and fast; the asc and pto
tiers cost roughly 20-40s per variant under the simulator, so a full sweep of
those is on the order of half an hour.

Exit status is nonzero if anything reported FAIL. XFAIL (a documented toolchain
limitation) and TODO (an unsolved puzzle) do not fail the run, but both are
counted and listed, so neither can hide.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIERS = ("torch", "asc", "pto")
KERNELS = ("cast_back", "per_token", "per_block", "per_channel")
STATUS_RE = re.compile(r"\[status\] variant=(\S+) result=(\w+) seconds=([\d.]+)(?: note=(.*))?")


def discover(tier: str | None, kernel: str | None, role: str) -> list[Path]:
    files: list[Path] = []
    for t in TIERS if tier is None else (tier,):
        for k in KERNELS if kernel is None else (kernel,):
            d = ROOT / "puzzles" / t / "quant" / role / k
            if d.is_dir():
                files.extend(sorted(d.glob("[0-9]*.py")))
    return files


def run_one(path: Path) -> dict:
    rel = path.relative_to(ROOT)
    start = time.time()
    proc = subprocess.run([sys.executable, "-B", str(path)], cwd=str(ROOT),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.time() - start
    match = None
    for line in proc.stdout.splitlines():
        m = STATUS_RE.match(line.strip())
        if m:
            match = m
    if match:
        return {"path": rel, "variant": match.group(1), "result": match.group(2),
                "seconds": float(match.group(3)), "note": match.group(4) or "",
                "wall": wall, "output": proc.stdout}
    return {"path": rel, "variant": str(rel), "result": "NO_STATUS",
            "seconds": 0.0, "note": f"exit={proc.returncode}", "wall": wall,
            "output": proc.stdout}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=TIERS)
    ap.add_argument("--kernel", choices=KERNELS)
    ap.add_argument("--role", choices=("answer", "puzzle"), default="answer")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel processes (keep at 1 for the simulator tiers)")
    ap.add_argument("--verbose", action="store_true", help="dump output of failures")
    args = ap.parse_args()

    files = discover(args.tier, args.kernel, args.role)
    if not files:
        print("nothing to run")
        return 1
    print(f"running {len(files)} variants ({args.role}, jobs={args.jobs})\n")

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run_one, files))
    else:
        results = []
        for f in files:
            r = run_one(f)
            results.append(r)
            print(f"  {r['result']:9} {r['wall']:6.1f}s  {r['variant']}")
        print()

    width = max(len(r["variant"]) for r in results)
    print(f"{'variant'.ljust(width)}  {'result':9} {'wall':>7}  note")
    print("-" * (width + 28))
    for r in results:
        print(f"{r['variant'].ljust(width)}  {r['result']:9} {r['wall']:6.1f}s  {r['note'][:60]}")

    tally: dict[str, int] = {}
    for r in results:
        tally[r["result"]] = tally.get(r["result"], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    total_wall = sum(r["wall"] for r in results)
    print(f"total wall time {total_wall / 60:.1f} min")

    bad = [r for r in results if r["result"] in ("FAIL", "NO_STATUS")]
    if args.verbose:
        for r in bad:
            print(f"\n===== output of {r['variant']} =====\n{r['output']}")
    if bad:
        print(f"\n{len(bad)} variant(s) failed:")
        for r in bad:
            print(f"  {r['variant']}: {r['note']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
