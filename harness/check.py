"""Run the ladder.

    python -m harness.check                      # every variant, every tier
    python -m harness.check asc                  # one tier
    python -m harness.check asc/per_token        # one kernel in one tier
    python -m harness.check asc/per_token/05     # one variant
    python -m harness.check per_token/05         # that variant in all three tiers
    python -m harness.check pto_05               # ambiguous -> lists the candidates
    python -m harness.check --role puzzle asc    # check the unsolved puzzles
    python -m harness.check --list               # print every id

A single id runs in this process, so the simulator hand-off and its output are
visible live. Several ids run one subprocess each -- the NPU tiers re-exec
themselves under ``msprof op simulator``, which has to be one variant per
process.

Exit status is nonzero only if something reported FAIL (or produced no verdict).
XFAIL (a documented toolchain limit) and TODO (an unwritten puzzle) are counted
and listed but do not fail the run, so neither can be mistaken for success and
neither masquerades as failure.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness.spec import TIERS, Variant, all_variants

ROOT = Path(__file__).resolve().parents[1]
STATUS_RE = re.compile(r"\[status\] variant=(\S+) result=(\w+) seconds=([\d.]+)(?: note=(.*))?")


# ---------------------------------------------------------------------------
# resolving a query to variants
# ---------------------------------------------------------------------------

def _token_matches(tok: str, variant: Variant, tier: str) -> bool:
    return (
        tier == tok
        or tier.startswith(tok)
        or variant.kernel == tok
        or variant.kernel.startswith(tok)
        or variant.num == tok
        or variant.stem == tok
        or variant.stem.startswith(tok)
    )


def _matches(query: str, variant: Variant, tier: str) -> bool:
    if query == variant.id(tier):
        return True
    toks = [t for t in query.split("/") if t]
    if len(toks) == 1 and "_" in toks[0] and not _token_matches(toks[0], variant, tier):
        # allow the `pto_05` shorthand by splitting a lone underscored token
        toks = [t for t in toks[0].split("_") if t]
    return all(_token_matches(t, variant, tier) for t in toks)


def resolve(queries: list[str]) -> list[tuple[Variant, str]]:
    everything = [(v, t) for v in all_variants() for t in v.tiers]
    if not queries:
        return everything
    out: list[tuple[Variant, str]] = []
    for q in queries:
        hits = [(v, t) for v, t in everything if _matches(q, v, t)]
        if not hits:
            raise SystemExit(f"no variant matches {q!r}. Try --list.")
        for hit in hits:
            if hit not in out:
                out.append(hit)
    return out


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def _run_subprocess(variant: Variant, tier: str, role: str) -> dict:
    vid = variant.id(tier)
    argv = [sys.executable, "-B", "-m", "harness.check", vid]
    if role != "answer":
        argv += ["--role", role]
    start = time.time()
    proc = subprocess.run(argv, cwd=str(ROOT), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    wall = time.time() - start
    last = None
    for line in proc.stdout.splitlines():
        m = STATUS_RE.match(line.strip())
        if m:
            last = m
    if last is None:
        return {"id": vid, "result": "NO_STATUS", "wall": wall,
                "note": f"exit={proc.returncode}", "output": proc.stdout}
    return {"id": vid, "result": last.group(2), "wall": wall,
            "note": last.group(4) or "", "output": proc.stdout}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.check",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="tier / kernel / variant selectors")
    ap.add_argument("--role", choices=("answer", "puzzle"), default="answer")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel processes; keep at 1 for the simulator tiers")
    ap.add_argument("--list", action="store_true", help="print every id and exit")
    ap.add_argument("--verbose", action="store_true", help="dump output of failures")
    args = ap.parse_args(argv)

    if args.list:
        for v, t in resolve([]):
            print(v.id(t))
        return 0

    selected = resolve(args.query)

    # One variant: run here, so the simulator hand-off streams to the terminal.
    if len(selected) == 1:
        from harness.runner import run_one
        variant, tier = selected[0]
        return run_one(variant, tier, args.role)

    print(f"running {len(selected)} variants ({args.role}, jobs={args.jobs})\n")
    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda st: _run_subprocess(*st, args.role), selected))
    else:
        results = []
        for variant, tier in selected:
            r = _run_subprocess(variant, tier, args.role)
            results.append(r)
            print(f"  {r['result']:9} {r['wall']:6.1f}s  {r['id']}")
        print()

    width = max(len(r["id"]) for r in results)
    print(f"{'variant'.ljust(width)}  {'result':9} {'wall':>7}  note")
    print("-" * (width + 28))
    for r in results:
        print(f"{r['id'].ljust(width)}  {r['result']:9} {r['wall']:6.1f}s  {r['note'][:60]}")

    tally: dict[str, int] = {}
    for r in results:
        tally[r["result"]] = tally.get(r["result"], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print(f"total wall time {sum(r['wall'] for r in results) / 60:.1f} min")

    bad = [r for r in results if r["result"] in ("FAIL", "NO_STATUS")]
    if args.verbose:
        for r in bad:
            print(f"\n===== {r['id']} =====\n{r['output']}")
    if bad:
        print(f"\n{len(bad)} variant(s) failed:")
        for r in bad:
            print(f"  {r['id']}: {r['note']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
