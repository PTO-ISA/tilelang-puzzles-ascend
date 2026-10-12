#!/usr/bin/env python3
"""Measure simulated VF cycles per variant, and tabulate ASC : PTO : SIMT.

    python tools/vf_cycles.py asc/cast_back/01 pto/cast_back/01 simt/cast_back/01
    python tools/vf_cycles.py cast_back/01        # all vector tiers of one variant
    python tools/vf_cycles.py --all               # every vector variant (slow)
    python tools/vf_cycles.py --table             # re-print the saved table

The method is ``doc/vf_cycles_from_msprof_simulator.md``: run the kernel under
``msprof op simulator``, then sum the ``cycles`` column of
``core0.veccore0_instr_exe.csv`` over the rows whose ``instr`` names a VF launch
-- ``VF_SIMT`` for a SIMT kernel, ``VF`` (or ``VF_SIMD``) for a SIMD one.
``cycles`` is already the total for its row and is never scaled by
``call_count``.

Two things this tool will not do:

* It will not key on the ``pipe`` column. Real VF rows carry ``pipe=PUSHQ``, and
  ``MOVEMASK`` / ``PUSH_PB`` share that lane, so ``instr`` is the only safe key.
* It will not report a cycle count for a variant that did not pass. A number
  from a kernel that computes the wrong answer is worse than no number, so the
  verdict is parsed from the same run and carried into the table.

Results are cached in ``.vf_cycles.json`` so the table can be rebuilt without
re-running the simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".vf_cycles.json"

# The launch-row names, lowercased. One kernel is one kind.
SIMT_NAMES = {"vf_simt"}
SIMD_NAMES = {"vf", "vf_simd"}

VECTOR_TIERS = ("asc", "pto", "simt")

OUT_RE = re.compile(r"^\[sim\] out=(.+)$", re.M)
STATUS_RE = re.compile(r"^\[status\] variant=(\S+) result=(\w+)", re.M)


# ---------------------------------------------------------------------------
# the CSV
# ---------------------------------------------------------------------------

def _instr_csv(out_dir: Path) -> Path | None:
    """The per-instruction CSV for core0.veccore0, which is the core we measure."""
    hits = sorted(out_dir.rglob("*instr_exe*.csv"))
    for p in hits:
        if "core0.veccore0" in p.name or "core0.veccore0" in str(p.parent):
            return p
    return hits[0] if hits else None


def sum_vf_cycles(out_dir: Path) -> dict:
    """Sum the VF launch rows. Reports what it found rather than guessing."""
    res = {"csv": None, "kind": None, "instr": None, "rows": 0,
           "call_count": 0, "cycles": None, "us": None, "seen": []}
    path = _instr_csv(out_dir)
    if path is None:
        return res
    res["csv"] = str(path)

    grouped: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0, 0])
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("instr") or "").strip()
            seen.add(name)
            if name.lower() not in SIMT_NAMES | SIMD_NAMES:
                continue
            try:
                g = grouped[name]
                g[0] += float(row.get("cycles") or 0)
                g[1] += float(row.get("running_time(us)") or 0)
                g[2] += int(float(row.get("call_count") or 0))
                g[3] += 1
            except ValueError:
                continue
    res["seen"] = sorted(seen)
    if not grouped:
        return res

    simt = {n: v for n, v in grouped.items() if n.lower() in SIMT_NAMES}
    simd = {n: v for n, v in grouped.items() if n.lower() in SIMD_NAMES}
    chosen, kind = (simt, "SIMT") if simt else (simd, "SIMD")
    if simt and simd:  # should not happen for one kernel; keep the bigger half
        if sum(v[0] for v in simd.values()) > sum(v[0] for v in simt.values()):
            chosen, kind = simd, "SIMD"
    res.update(kind=kind,
               instr=",".join(sorted(chosen)),
               cycles=sum(v[0] for v in chosen.values()),
               us=sum(v[1] for v in chosen.values()),
               call_count=sum(v[2] for v in chosen.values()),
               rows=sum(v[3] for v in chosen.values()))
    return res


# ---------------------------------------------------------------------------
# running one variant
# ---------------------------------------------------------------------------

def measure(vid: str) -> dict:
    """Run one variant under the simulator and sum its VF launch rows."""
    env = dict(os.environ, TLP_KEEP_SIM_OUT="1")
    proc = subprocess.run([sys.executable, "-B", "-m", "harness.check", vid],
                          cwd=str(ROOT), env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout
    rec = {"id": vid, "result": None, "cycles": None, "note": ""}

    m = STATUS_RE.search(out)
    rec["result"] = m.group(2) if m else "NO_STATUS"

    om = OUT_RE.search(out)
    if not om:
        rec["note"] = "no [sim] out= line (did the variant run under the simulator?)"
        return rec
    got = sum_vf_cycles(Path(om.group(1).strip()))
    rec.update({k: got[k] for k in ("csv", "kind", "instr", "rows",
                                    "call_count", "cycles", "us")})
    if got["cycles"] is None:
        # The doc's rule: read the instr column, do not invent a name.
        rec["note"] = ("no VF launch row; instr values present: "
                       + ", ".join(got["seen"][:16]))
    elif rec["result"] != "PASS":
        rec["note"] = f"result={rec['result']} -- cycles withheld from the table"
    return rec


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

def build_table(records: dict) -> str:
    """ASC : PTO : SIMT per variant, with ASC as the 1.00 baseline."""
    by_variant: dict[str, dict[str, dict]] = defaultdict(dict)
    for vid, rec in records.items():
        tier, kernel, stem = vid.split("/")
        by_variant[f"{kernel}/{stem}"][tier] = rec

    lines = [
        f"{'variant':34} {'ASC':>8} {'PTO':>8} {'SIMT':>9}   "
        f"{'PTO/ASC':>8} {'SIMT/ASC':>9}",
        "-" * 92,
    ]
    ratios: dict[str, list[float]] = {"pto": [], "simt": []}
    for name in sorted(by_variant):
        row = by_variant[name]

        def cyc(t: str):
            r = row.get(t)
            if not r or r.get("cycles") is None:
                return None
            return None if r.get("result") != "PASS" else r["cycles"]

        a, p, s = cyc("asc"), cyc("pto"), cyc("simt")
        f = lambda v: "-" if v is None else f"{v:,.0f}"
        rp = f"{p / a:.2f}x" if a and p else "-"
        rs = f"{s / a:.2f}x" if a and s else "-"
        if a and p:
            ratios["pto"].append(p / a)
        if a and s:
            ratios["simt"].append(s / a)
        lines.append(f"{name:34} {f(a):>8} {f(p):>8} {f(s):>9}   {rp:>8} {rs:>9}")

    lines.append("-" * 92)
    for tier in ("pto", "simt"):
        v = ratios[tier]
        if v:
            n = len(v)
            lines.append(f"{tier.upper()} vs ASC over {n} variant"
                         f"{'' if n == 1 else 's'}: "
                         f"{min(v):.2f}x .. {max(v):.2f}x, mean {sum(v) / n:.2f}x")
    lines += [
        "",
        "Simulated VF cycles on core0.veccore0 (soc Ascend950PR_9599), summed over",
        "the VF launch rows of *_instr_exe.csv. ASC is the 1.00 baseline; a ratio",
        "below 1.00 means that tier used FEWER simulated cycles than ASC.",
        "A dash means the variant did not pass, or produced no VF launch row --",
        "cycles are never reported for a kernel that computed the wrong answer.",
        "These are simulator numbers; no silicon-fidelity claim is made.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="variant ids or selectors")
    ap.add_argument("--all", action="store_true", help="every vector-tier variant")
    ap.add_argument("--table", action="store_true", help="print the cached table only")
    args = ap.parse_args()

    records: dict = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    if args.table:
        print(build_table(records))
        return 0

    sys.path.insert(0, str(ROOT))
    from harness.check import resolve                      # noqa: E402

    if args.all:
        ids = [v.id(t) for v, t in resolve([]) if t in VECTOR_TIERS]
    elif args.query:
        ids = [v.id(t) for v, t in resolve(args.query) if t in VECTOR_TIERS]
    else:
        ap.error("give one or more selectors, or --all")

    if not ids:
        print("no vector-tier variants matched (torch has no simulator run)")
        return 1

    print(f"measuring {len(ids)} variant(s) under the simulator\n")
    for vid in ids:
        rec = measure(vid)
        records[vid] = rec
        cyc = "-" if rec.get("cycles") is None else f"{rec['cycles']:,.0f}"
        extra = f"  {rec['note']}" if rec.get("note") else ""
        print(f"  {vid:34} {rec['result']:>8} {rec.get('instr') or '-':>8} "
              f"{cyc:>10} cycles  calls={rec.get('call_count', 0)}{extra}")
        CACHE.write_text(json.dumps(records, indent=1, sort_keys=True))

    print()
    print(build_table(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
