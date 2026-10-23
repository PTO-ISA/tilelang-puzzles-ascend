"""Audit the msprof traces: is the profiled kernel actually the quant kernel?

    python tools/vf_trace_audit.py              # every measured id
    python tools/vf_trace_audit.py per_token    # a selector
    python tools/vf_trace_audit.py --hist asc/cast_back/01   # full histogram

Why this exists. ``msprof op simulator`` profiles a **single operator** -- in
practice the first one the process launches -- so a cycle count is only the
kernel's if the kernel is what got profiled. That is not a theoretical worry:
``per_token/06`` and ``per_block/05`` were originally measured on the
``torch.zeros`` fill their ``launch()`` runs first, and reported identical
cycles in all three tiers because all three had measured the same aten kernel.

The cycle CSV cannot tell you what it profiled -- its VF row is one aggregated
launch. The simulator's instruction log can: it names every instruction that
issued, including the vector pipes (``RVECEX`` / ``RVECLD`` / ``RVECST`` /
``RVECSU`` on the SIMD tiers, the ``SIMT_*`` stream on the SIMT tier). This
reads that log and checks the mnemonics are the ones a quantization kernel is
made of.

The contrast is stark enough to settle it. The zero-fill that was measured by
mistake issues ``RV_VDUPS`` and two stores and nothing else -- no convert, no
multiply, no maximum. A real quant kernel cannot look like that, because
*converting* and *scaling* are what it does.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".vf_cycles.json"

# A vector-pipe instruction line in core0.veccore0.instr_log.dump.
LINE = re.compile(r"\) (RVEC\w+|VEC|PUSHQ) *:.*?\(ID: \d+\) ([A-Za-z0-9_]+)")

# Issue/sync traffic rather than work on data.
SYNC = {"RV_SEND", "SET_FLAG", "WAIT_FLAG", "RV_SMEM_BAR", "SIMT_END",
        "SIMT_BRANCH", "VF", "VF_SIMT", "VF_SIMD", "RV_SMOV", "RV_SMOVI",
        "SIMT_S2R", "SIMT_MOVI"}

# The instruction families a quantization kernel is built from. A trace missing
# the first two is not a quant kernel, whatever else it contains.
FAMILIES = {
    "convert": ("CVT", "F2F", "F2I", "I2F"),
    "scale": ("VMUL", "FMUL", "VDIV", "FFMA", "SIMT_FMUL"),
    "reduce/abs": ("VMAX", "VCMAX", "VCGMAX", "VABS", "FMNMX", "VMIN"),
    "bit math": ("VAND", "VOR", "VSHL", "VSHR", "VSUB", "VADD", "IADD",
                 "LOP3", "SHFI", "ISETP", "VXOR"),
    "lane moves": ("VSEL", "VDUPS", "VINTLV", "VUNZIP", "VGATHER", "VBRC",
                   "PUNPACK", "SHFL"),
    "memory": ("VLD", "VST", "LDS", "STS", "LDG", "STG", "LEA"),
}


def histogram(dump: Path) -> collections.Counter:
    hist: collections.Counter = collections.Counter()
    with dump.open(errors="replace") as fh:
        for line in fh:
            m = LINE.search(line)
            if m:
                hist[m.group(2)] += 1
    return hist


def families_present(hist: collections.Counter) -> dict[str, int]:
    out = {}
    for fam, keys in FAMILIES.items():
        out[fam] = sum(n for mn, n in hist.items()
                       if mn not in SYNC and any(k in mn for k in keys))
    return out


def classify(fam: dict[str, int]) -> tuple[bool, str]:
    """Does this instruction mix belong to a quantization kernel?

    The test is **multiply, plus either a format conversion or a maximum**.

    Every kernel in this ladder multiplies by a scale; that is what makes it a
    quantization kernel rather than a copy or a fill. It then either converts
    between formats (the `cast_back` family) or computes an amax (everything
    with a reduction), and usually both.

    An earlier version of this required a *convert*, and flagged
    `simt/per_token/04` -- wrongly. That variant takes float32 input, so it
    needs no widening convert, and writes hand-packed FP4 nibbles, so it has no
    narrowing convert either: its format work is integer bit math. Zero
    converts is what its source predicts, which makes the trace more convincing
    rather than less.

    The negative control is the `torch.zeros` fill this audit exists to catch:
    one `RV_VDUPS` and two stores, so no multiply at all.
    """
    if fam["scale"] == 0:
        return False, "** NO MULTIPLY -- not a quant kernel **"
    if fam["convert"] == 0 and fam["reduce/abs"] == 0:
        return False, "** no convert and no reduction **"
    if fam["convert"] == 0:
        return True, "quant kernel (bit-math format, no convert)"
    return True, "quant kernel"


def dump_for(csv_path: str) -> Path | None:
    # .../OPPROF_x/simulator/core0.veccore0/core0.veccore0_instr_exe.csv
    opprof = Path(csv_path).parents[2]
    p = opprof / "dump" / "core0.veccore0.instr_log.dump"
    return p if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="substring filter on the variant id")
    ap.add_argument("--hist", metavar="ID",
                    help="print the full mnemonic histogram for one id")
    ap.add_argument("--dump", metavar="PATH",
                    help="audit an arbitrary msprof OPPROF_* directory -- used to "
                         "re-check the negative control (a torch.zeros fill)")
    args = ap.parse_args()

    if not CACHE.exists():
        print("no .vf_cycles.json; run tools/vf_cycles.py --all first")
        return 2
    records = json.loads(CACHE.read_text())

    if args.dump:
        root = Path(args.dump)
        d = root / "dump" / "core0.veccore0.instr_log.dump"
        if not d.exists():
            d = root
        if not d.exists():
            print(f"no instruction log at {d}")
            return 2
        hist = histogram(d)
        fam = families_present(hist)
        ok, verdict = classify(fam)
        print(f"{d}\n")
        for mn, n in hist.most_common(20):
            print(f"  {mn:20} {n:>7}{'  (sync)' if mn in SYNC else ''}")
        print()
        for k, v in fam.items():
            print(f"  {k:12} {v:>6}")
        print(f"\n  verdict: {verdict}")
        return 0 if ok else 1

    if args.hist:
        hits = [k for k in records if args.hist in k]
        if len(hits) != 1:
            print(f"--hist needs exactly one match, got {len(hits)}: {hits[:5]}")
            return 2
        vid = hits[0]
        dump = dump_for(records[vid]["csv"])
        if dump is None:
            print(f"{vid}: the simulator output has been cleaned up; re-measure it")
            return 1
        hist = histogram(dump)
        calls = records[vid].get("call_count") or 1
        print(f"{vid}  ({calls} VF launches, "
              f"{records[vid].get('cycles'):,.0f} cycles)\n")
        print(f"  {'mnemonic':20} {'total':>7} {'per launch':>11}")
        for mn, n in hist.most_common():
            tag = "  (sync)" if mn in SYNC else ""
            print(f"  {mn:20} {n:>7} {n / calls:>11.2f}{tag}")
        print()
        for fam, n in families_present(hist).items():
            print(f"  {fam:12} {n:>6} instructions")
        return 0

    ids = sorted(k for k in records
                 if not args.query or any(q in k for q in args.query))
    print(f"auditing {len(ids)} trace(s) for quant-kernel instruction families\n")
    hdr = (f"{'variant':38} {'launches':>8} {'cvt':>6} {'scale':>6} {'red':>6} "
           f"{'bits':>6} {'lanes':>6} {'mem':>6}  verdict")
    print(hdr)
    print("-" * len(hdr))

    bad, gone = [], []
    for vid in ids:
        rec = records[vid]
        dump = dump_for(rec.get("csv", ""))
        if dump is None:
            gone.append(vid)
            print(f"{vid:38} {'':>8} {'output cleaned up -- re-measure to audit':>6}")
            continue
        hist = histogram(dump)
        fam = families_present(hist)
        calls = rec.get("call_count") or 1
        ok, verdict = classify(fam)
        if not ok:
            bad.append(vid)
        print(f"{vid:38} {calls:>8} {fam['convert']:>6} {fam['scale']:>6} "
              f"{fam['reduce/abs']:>6} {fam['bit math']:>6} {fam['lane moves']:>6} "
              f"{fam['memory']:>6}  {verdict}")

    print()
    print("cvt = converts, scale = multiply/divide, red = max/abs reductions,")
    print("bits = integer bit math, lanes = select/broadcast/interleave/gather,")
    print("mem = vector loads and stores. Counts are whole-trace, not per launch.")
    print()
    if gone:
        print(f"{len(gone)} trace(s) no longer on disk (the simulator output is a")
        print("cache); re-run tools/vf_cycles.py for those ids to audit them.")
    if bad:
        print(f"{len(bad)} trace(s) DO NOT look like quant kernels: "
              f"{', '.join(bad)}")
        print("Their cycle numbers must not be published. See the module docstring.")
        return 1
    print("Every audited trace converts and scales/reduces -- the instruction mix")
    print("is the quant kernel's, not an unrelated operator's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
