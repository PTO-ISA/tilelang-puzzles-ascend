"""Measure each kernel's T.SimdVF body: lines of code and vector operations.

The PTO-vs-ASC argument rests on VMI needing fewer vector *operations* for the
same work. This measures that instead of asserting it.

Two numbers, because they disagree and the disagreement is informative:

**ops** counts calls to the vector intrinsics (S.* / V.*). This is the metric the
claim is actually about -- how many machine operations the backend forces you to
spell out.

**lines** counts non-comment source lines. VMI needs an explicit `size=` on nearly
every call and a mask on many, so its lines are *wider*; on variants where it
saves no operations it can come out slightly longer. Reporting only lines would
flatter ASC, and reporting only ops would hide a real ergonomic cost of VMI.

Caveat on magnitude: these teaching kernels are single-config, so each one spells
the broadcast/select/pack machinery once. Production kernels branch over
round_sf, packing, FP4, column-major and requant, which repeats that machinery
per branch -- which is why the production port (TileKernels 5395526) came to 83
net lines removed across four kernels, a much larger relative saving than
anything visible here.

    python tools/vf_lines.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


OP_RE = re.compile(r"\b[SV]\.(\w+)\s*\(")
# Not vector machine operations, so excluded from the op count:
#   - allocation and type constructors
#   - mask construction (predicate setup, not data movement or arithmetic)
#   - bit reinterpretation, which emits no instruction at all. This one matters
#     for fairness: ASC spells it T.reinterpret (outside this regex) while VMI
#     spells it V.vinterpret_cast (inside it), so counting the VMI form would
#     penalise PTO for a purely notational difference.
NOT_OPS = {
    "alloc_local", "alloc_var", "vreg",
    "create_mask", "pset", "pnot",
    "vinterpret_cast",
}


def vf_body(path: Path) -> tuple[int, int]:
    """Return (lines, vector-op calls) inside `with T.SimdVF():` blocks."""
    lines = path.read_text().splitlines()
    # skip the module docstring, which also mentions T.SimdVF
    start = 0
    if lines and lines[0].lstrip().startswith('"""'):
        for i in range(1, len(lines)):
            if '"""' in lines[i]:
                start = i + 1
                break
    total = ops = 0
    i = start
    while i < len(lines):
        m = re.match(r"^(\s*)with T\.SimdVF\(", lines[i])
        if not m:
            i += 1
            continue
        indent = len(m.group(1))
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            if line.strip() and not line.strip().startswith("#"):
                total += 1
                ops += sum(1 for n in OP_RE.findall(line) if n not in NOT_OPS)
            i += 1
    return total, ops


def main() -> int:
    rows = []
    for kernel in ("cast_back", "per_token", "per_block", "per_channel"):
        for variant in sorted((ROOT / "puzzles/asc/quant/answer" / kernel).glob("[0-9]*.py")):
            pto = ROOT / "puzzles/pto/quant/answer" / kernel / variant.name
            if not pto.exists():
                continue
            rows.append((f"{kernel}/{variant.stem}", vf_body(variant), vf_body(pto)))
    if not rows:
        print("no paired asc/pto variants yet")
        return 0
    w = max(len(r[0]) for r in rows)
    head = (f"{'variant'.ljust(w)}  {'ops ASC':>8} {'ops PTO':>8} {'d':>5}   "
            f"{'ln ASC':>7} {'ln PTO':>7} {'d':>5}")
    print(head)
    print("-" * len(head))
    tla = tlp = toa = top = 0
    for name, (la, oa), (lp, op) in rows:
        tla += la; tlp += lp; toa += oa; top += op
        print(f"{name.ljust(w)}  {oa:8} {op:8} {op - oa:+5}   "
              f"{la:7} {lp:7} {lp - la:+5}")
    print("-" * len(head))
    print(f"{'total'.ljust(w)}  {toa:8} {top:8} {top - toa:+5}   "
          f"{tla:7} {tlp:7} {tlp - tla:+5}")
    if toa:
        print(f"\nPTO vector operations relative to ASC: {top / toa:.0%}")
    if tla:
        print(f"PTO source lines relative to ASC:      {tlp / tla:.0%}")
    print("\nOps is the metric the PTO-vs-ASC claim is about. Lines runs the other")
    print("way on variants where VMI saves no operations, because size= and mask")
    print("arguments make each call wider -- a real ergonomic cost, worth seeing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
