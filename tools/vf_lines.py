"""Measure each kernel's VF body across the three vector tiers.

Three columns, one per vector tier, plus the measured cycle counts when
``tools/vf_cycles.py`` has been run.

**lines** counts non-comment source lines inside the VF region. This is the one
number that means the same thing in all three vector tiers, so it carries the
"how much is there to write" comparison.

**ops** counts calls to the vector intrinsics (``S.*`` / ``V.*``) -- how many
machine operations the backend forces you to spell out. It exists for ASC and
PTO only, and the SIMT column is deliberately a dash rather than a number:
SIMT's arithmetic is plain operators on scalars (``*``, ``>>``, ``&``), so the
backend chooses the instructions and there is nothing in the source to count.
Printing some ``T.*`` call count beside the SIMD op counts would invite a
comparison between quantities that are not the same thing.

Why both, for ASC vs PTO: they disagree, and the disagreement is informative.
VMI needs an explicit ``size=`` on nearly every call and a mask on many, so its
lines are *wider*; on variants where it saves no operations it can come out
slightly longer. Reporting only lines would flatter ASC, and reporting only ops
would hide a real ergonomic cost of VMI.

Caveat on scope: only the VF region is counted, so a helper defined outside it
is invisible in every tier -- PTO's ``@T.macro`` bodies and SIMT's plain Python
expression builders alike. That is symmetric but not harmless: at
``per_token/04`` and ``per_block/03`` PTO's excluded helper is a five-line
round-to-odd, while SIMT's is an eighteen-line hand-written e2m1 encoder,
because no FP4 convert is reachable from a SimtVF region. Those two rows
understate SIMT's real cost; the variant pages say so.

Second scope rule, for the same reason: where a variant ships an *alternative
spelling* -- a second kernel in the same file, behind a trace-time flag -- the
alternative is bracketed with ``# --- BEGIN ALT`` / ``# --- END ALT`` and no
line between them is counted, VF regions included. Only one of the two is ever
emitted, and the column means "the tier's own kernel". Today that is only
``simt/per_token/04``'s ``compile_hybrid_kernel``, which leaves the SimtVF
region for the FP4 convert.

Caveat on magnitude: these teaching kernels are single-config, so each one
spells the broadcast/select/pack machinery once. Production kernels branch over
round_sf, packing, FP4, column-major and requant, which repeats that machinery
per branch -- which is why the production port (TileKernels-PTO 5395526) came to
83 net lines removed across four kernels, a much larger relative saving than
anything visible here.

Second caveat, and the reason the cycles column is here: all the counts above
are STATIC -- operations written, not operations executed. Source length and
runtime cost are different axes, and on this ladder they point in opposite
directions more often than not.

    python tools/vf_lines.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CYCLES = ROOT / ".vf_cycles.json"

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

# tier -> (column label, the `with` opener that starts a VF region, op regex or
# None when the tier has no countable vector intrinsics)
TIERS = {
    "asc": ("ASC", "with T.SimdVF(", OP_RE),
    "pto": ("PTO", "with T.SimdVF(", OP_RE),
    "simt": ("SIMT", "with T.SimtVF(", None),
}

KERNELS = ("cast_back", "per_token", "per_block", "per_channel")


def vf_body(path: Path, opener: str, op_re: re.Pattern | None) -> tuple[int, int | None]:
    """Return (lines, vector-op calls) inside the tier's VF regions."""
    lines = path.read_text().splitlines()
    # Drop any ALTERNATIVE spelling first: a second kernel in the same file,
    # behind a trace-time flag, of which only one is ever emitted. The published
    # counts are of the tier's own kernel. See the scope rules in the docstring.
    kept, skipping = [], False
    for line in lines:
        if "--- BEGIN ALT" in line:
            skipping = True
        elif "--- END ALT" in line:
            skipping = False
        elif not skipping:
            kept.append(line)
    lines = kept
    # skip the module docstring, which also mentions the opener
    start = 0
    if lines and lines[0].lstrip().startswith('"""'):
        for i in range(1, len(lines)):
            if '"""' in lines[i]:
                start = i + 1
                break
    opener_re = re.compile(r"^(\s*)" + re.escape(opener))
    total = ops = 0
    i = start
    while i < len(lines):
        m = opener_re.match(lines[i])
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
                if op_re is not None:
                    ops += sum(1 for n in op_re.findall(line) if n not in NOT_OPS)
            i += 1
    return total, (ops if op_re is not None else None)


def _cycles() -> dict:
    """variant -> {tier: cycles}, PASS rows only. Empty if never measured."""
    if not CYCLES.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    for vid, rec in json.loads(CYCLES.read_text()).items():
        tier, kernel, stem = vid.split("/")
        if rec.get("result") == "PASS" and rec.get("cycles") is not None:
            out.setdefault(f"{kernel}/{stem}", {})[tier] = rec["cycles"]
    return out


def main() -> int:
    cyc = _cycles()
    rows = []
    for kernel in KERNELS:
        base = ROOT / "puzzles/asc/quant/answer" / kernel
        for variant in sorted(base.glob("[0-9]*.py")):
            per_tier = {}
            for tier, (_, opener, op_re) in TIERS.items():
                path = ROOT / "puzzles" / tier / "quant/answer" / kernel / variant.name
                per_tier[tier] = vf_body(path, opener, op_re) if path.exists() else None
            rows.append((f"{kernel}/{variant.stem}", per_tier))
    if not rows:
        print("no variants found")
        return 0

    w = max(len(r[0]) for r in rows)
    head = (f"{'variant'.ljust(w)}  {'ops ASC':>7} {'ops PTO':>7}  "
            f"{'ln ASC':>6} {'ln PTO':>6} {'ln SIMT':>7}  "
            f"{'cyc ASC':>8} {'cyc PTO':>8} {'cyc SIMT':>9}")
    print(head)
    print("-" * len(head))

    tot = {t: [0, 0] for t in TIERS}
    n_simt = 0
    for name, per_tier in rows:
        def ln(t):
            return per_tier[t][0] if per_tier[t] else None
        def op(t):
            return per_tier[t][1] if per_tier[t] else None
        for t in TIERS:
            if per_tier[t]:
                tot[t][0] += per_tier[t][0]
                tot[t][1] += per_tier[t][1] or 0
        if per_tier["simt"]:
            n_simt += 1
        c = cyc.get(name, {})
        f = lambda v: "-" if v is None else f"{v:,.0f}"
        print(f"{name.ljust(w)}  {f(op('asc')):>7} {f(op('pto')):>7}  "
              f"{f(ln('asc')):>6} {f(ln('pto')):>6} {f(ln('simt')):>7}  "
              f"{f(c.get('asc')):>8} {f(c.get('pto')):>8} {f(c.get('simt')):>9}")

    print("-" * len(head))
    print(f"{'total'.ljust(w)}  {tot['asc'][1]:>7} {tot['pto'][1]:>7}  "
          f"{tot['asc'][0]:>6} {tot['pto'][0]:>6} {tot['simt'][0]:>7}")
    if tot["asc"][1]:
        print(f"\nPTO vector operations relative to ASC: "
              f"{tot['pto'][1] / tot['asc'][1]:.0%}")
    if tot["asc"][0]:
        print(f"PTO source lines relative to ASC:      "
              f"{tot['pto'][0] / tot['asc'][0]:.0%}")
        if n_simt == len(rows):
            print(f"SIMT source lines relative to ASC:     "
                  f"{tot['simt'][0] / tot['asc'][0]:.0%}")
        elif n_simt:
            print(f"SIMT source lines: {n_simt}/{len(rows)} variants written; "
                  f"the ratio is reported once the tier is complete.")

    print("\nOps is the metric the PTO-vs-ASC claim is about. Lines runs the other")
    print("way on variants where VMI saves no operations, because size= and mask")
    print("arguments make each call wider -- a real ergonomic cost, worth seeing.")
    print("SIMT has no ops column by design: its arithmetic is plain operators on")
    print("scalars, so the backend picks the instructions and the source has")
    print("nothing comparable to count.")
    print()
    print("Helpers outside the VF region are not counted, in any tier. That hides")
    print("real work at per_token/04 and per_block/03, where SIMT hand-writes an")
    print("18-line e2m1 encoder against PTO's 5-line round-to-odd macro -- those")
    print("two SIMT line counts are understated.")
    print()
    print("The first six columns are STATIC -- operations and lines written, not")
    print("executed. Where PTO uses a wider vector it also runs fewer loop")
    print("iterations, and that does not show up in a static count. per_block is")
    print("the clear case: PTO's static count is slightly higher, but its")
    print("reduction loop is 8 iterations of 128 lanes against ASC's 16 of 64.")
    print()
    print("The cycles columns are MEASURED, from tools/vf_cycles.py (simulated VF")
    print("cycles, PASS rows only; a blank means not measured yet). Read them")
    print("against the line counts rather than instead of them: shorter source")
    print("and fewer cycles are different axes, and on the SIMT column they")
    print("point in opposite directions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
