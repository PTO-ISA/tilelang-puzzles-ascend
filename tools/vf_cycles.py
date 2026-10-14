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

# Variants whose launch() runs a device op (a torch.zeros fill for an input the
# unused modes never read) BEFORE their own kernel. msprof op simulator profiles
# only the first operator, so for these the ordinary path measures that fill --
# identically in all three tiers, which is how it was caught. They are measured
# through tools/vf_drive.py instead, which makes the kernel the first op and
# names the mode. Value is the mode driven.
DRIVEN = {
    "per_token/06_split_requant": "full",
    "per_block/05_split_compose": "full",
}

# Alternative spellings of a variant that are shipped and verified, but are not
# a variant of their own: the harness entry point exposes one of them as the
# default, so the other needs the driver. Reported in their own section rather
# than in the 22-row ASC:PTO:SIMT table, which stays one row per variant.
# id -> (driver mode, what it is)
EXTRA = {
    "simt/per_token/04_fp32_in_fp4_out": [
        ("branch_free", "the e2m1 encoder's conditionals as mask arithmetic "
                        "instead of `T.if_then_else`"),
        ("hw_convert", "no encoder at all: the kernel leaves the SimtVF region "
                       "for a three-line `T.SimdVF` block holding the one "
                       "instruction the thread model cannot name"),
    ],
}
EXTRA_SEP = "+"

OUT_RE = re.compile(r"^\[sim\] out=(.+)$", re.M)
STATUS_RE = re.compile(r"^\[status\] variant=(\S+) result=(\w+)", re.M)
SOC_RE = re.compile(r"soc=(\S+?)[,)\s]", re.M)


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
    if simt and simd:
        # A HYBRID kernel: one that opens a T.SimtVF region and a T.SimdVF
        # region, which `simt/per_token/04` does in its hw_convert spelling to
        # reach the FP4 convert. Its cost is the sum of both launch kinds, not
        # the larger of them -- both run, on the same core, for the same token.
        chosen, kind = {**simt, **simd}, "SIMT+SIMD"
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

def measure_driven(vid: str, mode: str, prior: dict | None) -> dict:
    """Measure one variant through tools/vf_drive.py, keeping its harness verdict.

    The driver does not check numerics -- only ``python -m harness.check`` is
    allowed to say a variant is correct -- so this refuses to report cycles
    unless a prior harness run of the same id PASSed.
    """
    tier, kernel, stem = vid.split("/")
    short = f"{tier}/{kernel}/{stem.split('_')[0]}"
    env = dict(os.environ, TLP_KEEP_SIM_OUT="1")
    proc = subprocess.run([sys.executable, "-B", str(ROOT / "tools/vf_drive.py"),
                           short, mode],
                          cwd=str(ROOT), env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    rec = {"id": vid, "method": "driver", "mode": mode,
           "result": (prior or {}).get("result"), "cycles": None, "note": ""}
    if rec["result"] != "PASS":
        rec["note"] = ("no PASS from `python -m harness.check` for this id, so "
                       "driven cycles are withheld")
        return rec
    om = OUT_RE.search(proc.stdout)
    if not om:
        rec["note"] = "driver produced no [sim] out= line"
        return rec
    got = sum_vf_cycles(Path(om.group(1).strip()))
    rec.update({k: got[k] for k in ("csv", "kind", "instr", "rows",
                                    "call_count", "cycles", "us")})
    soc = SOC_RE.search(proc.stdout)
    rec["soc"] = soc.group(1) if soc else (prior or {}).get("soc")
    rec["shape"] = (prior or {}).get("shape")
    if got["cycles"] is None:
        rec["note"] = ("no VF launch row; instr values present: "
                       + ", ".join(got["seen"][:16]))
    else:
        rec["note"] = (f"measured through tools/vf_drive.py at mode={mode}; "
                       f"the harness path profiles this variant's placeholder "
                       f"torch.zeros fill instead of its kernel")
    return rec


def effective_shape(variant, tier: str) -> str | None:
    """The (M, K) this variant actually runs at, from the spec -- not the output.

    Parsing ``shape=(m, k)`` out of the relayed log looked fine and was wrong:
    it matched the *configured* base shape, so the three variants with their own
    shape rules (``per_token/07`` at K=256, ``per_channel/02`` and ``/04`` at
    M=64) were all recorded as 32x128. Ask the spec instead.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from harness import sim as _sim
        m, k = variant.shape.resolve(*_sim.sim_shapes())
        return f"{m}x{k}"
    except Exception:                                         # noqa: BLE001
        return None


def measure(vid: str, shape: str | None = None) -> dict:
    """Run one variant under the simulator and sum its VF launch rows."""
    env = dict(os.environ, TLP_KEEP_SIM_OUT="1")
    proc = subprocess.run([sys.executable, "-B", "-m", "harness.check", vid],
                          cwd=str(ROOT), env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout
    rec = {"id": vid, "method": "harness", "result": None, "cycles": None,
           "note": ""}

    m = STATUS_RE.search(out)
    rec["result"] = m.group(2) if m else "NO_STATUS"

    # Provenance, so the committed table can state the conditions it was taken
    # under rather than asking the reader to trust them.
    soc = SOC_RE.search(out)
    rec["soc"] = soc.group(1) if soc else None
    rec["shape"] = shape

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

def suspect_identical(by_variant: dict) -> list[str]:
    """Variants whose tiers report the SAME cycle count.

    Three different implementations cannot cost exactly the same, so an exact
    tie across tiers means the profiler captured something that is not the
    kernel. This is a guard against repeating a mistake already made once.
    """
    out = []
    for name, row in by_variant.items():
        vals = [r["cycles"] for t, r in row.items()
                if t in VECTOR_TIERS and r.get("result") == "PASS"
                and r.get("cycles") is not None]
        if len(vals) >= 2 and len(set(vals)) == 1:
            out.append(name)
    return sorted(out)


def build_table(records: dict) -> str:
    """ASC : PTO : SIMT per variant, with ASC as the 1.00 baseline."""
    by_variant: dict[str, dict[str, dict]] = defaultdict(dict)
    for vid, rec in records.items():
        if EXTRA_SEP in vid:
            continue                    # an alternative spelling, not a variant
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
    bad = suspect_identical(by_variant)
    if bad:
        lines.append("!! identical cycles across tiers (NOT a real measurement): "
                     + ", ".join(bad))
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


# Below this ratio a tier is called faster than ASC. Anything between it and
# 1.00 is reported as a draw: these are simulated cycles on one shape, and a
# fraction of a percent is not a result.
FASTER = 0.98

DATA = ROOT / "perf" / "quant_vf_cycles.json"
REPORT = ROOT / "doc" / "quant_vf_cycle_comparison.md"


def write_report(records: dict) -> None:
    """Write the tracked data file and the markdown summary.

    The cache (`.vf_cycles.json`) is scratch and holds absolute simulator paths
    that differ per run; this writes a committed copy with those stripped, so
    the published numbers are reviewable in a diff.
    """
    keep = ("result", "kind", "instr", "cycles", "call_count", "us", "soc",
            "shape", "method", "mode", "note")
    data = {vid: {k: rec.get(k) for k in keep if rec.get(k) not in (None, "")}
            for vid, rec in sorted(records.items())}
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")

    by_variant: dict[str, dict[str, dict]] = defaultdict(dict)
    for vid, rec in records.items():
        if EXTRA_SEP in vid:
            continue                    # an alternative spelling, not a variant
        tier, kernel, stem = vid.split("/")
        by_variant[f"{kernel}/{stem}"][tier] = rec

    socs = {r.get("soc") for r in records.values() if r.get("soc")}
    shapes = sorted({r.get("shape") for r in records.values() if r.get("shape")})

    def cyc(row, t):
        r = row.get(t)
        if not r or r.get("cycles") is None or r.get("result") != "PASS":
            return None
        return r["cycles"]

    lines = [
        "# Simulated VF cycles — ASC vs PTO vs SIMT",
        "",
        "Generated by `python tools/vf_cycles.py --all --report`. The raw numbers",
        "are in [`../perf/quant_vf_cycles.json`](../perf/quant_vf_cycles.json),",
        "committed alongside this file so the table can be checked in a diff.",
        "",
        "**These are simulator cycles, not hardware measurements.** The method is",
        "[`vf_cycles_from_msprof_simulator.md`](vf_cycles_from_msprof_simulator.md):",
        "run the kernel under `msprof op simulator`, then sum the `cycles` column",
        "of `core0.veccore0_instr_exe.csv` over the rows whose `instr` names a VF",
        "launch — `VF_SIMT` for a SIMT kernel, `VF` for a SIMD one. `cycles` is",
        "already the total for its row and is never scaled by `call_count`. No",
        "silicon-fidelity claim is made.",
        "",
        f"SoC `{'`, `'.join(sorted(socs)) or 'unrecorded'}`; "
        f"core `core0.veccore0`.",
        "",
        "**Each row's three tiers run at the same shape**, so the ratios are",
        "within-variant and valid; cycle counts are *not* comparable between rows,",
        "which do different amounts of work and in three cases run at a different",
        f"shape. Shapes in use: `{'`, `'.join(shapes) or 'unrecorded'}` (M x K).",
        "",
        "Every row below passed the same correctness check as its ASC and PTO",
        "twin — the same oracle comparison, the same `assert_on_device` proof that",
        "the result came back from the device. **Cycles are withheld for any",
        "variant that did not pass**: a number from a kernel that computes the",
        "wrong answer is worse than no number.",
        "",
        "| variant | ASC | PTO | SIMT | PTO/ASC | SIMT/ASC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ratios = {"pto": [], "simt": []}
    for name in sorted(by_variant):
        row = by_variant[name]
        a, pt, st = cyc(row, "asc"), cyc(row, "pto"), cyc(row, "simt")
        f = lambda v: "—" if v is None else f"{v:,.0f}"
        rp = f"{pt / a:.2f}x" if a and pt else "—"
        rs = f"{st / a:.2f}x" if a and st else "—"
        if a and pt:
            ratios["pto"].append((pt / a, name))
        if a and st:
            ratios["simt"].append((st / a, name))
        lines.append(f"| `{name}` | {f(a)} | {f(pt)} | {f(st)} | "
                     f"{'**' + rp + '**' if a and pt and pt / a < FASTER else rp}"
                     f" | {rs} |")

    odd = sorted({(n, r[t]["shape"]) for n, row in by_variant.items()
                  for t in row for r in [row]
                  if row[t].get("shape") and row[t]["shape"] != "32x128"})
    if odd:
        lines += ["", "Variants that do not run at the default `32x128`: "
                  + ", ".join(f"`{n}` at `{sh}`" for n, sh in odd) + "."]
    lines += ["", f"ASC is the 1.00 baseline. A ratio below {FASTER:.2f}x — shown in",
              "bold — means that tier used **fewer** simulated cycles than ASC.",
              "Ratios between that and 1.00x are draws: these are simulated cycles",
              "on one shape per variant, and a fraction of a percent is not a result.", ""]

    for tier in ("pto", "simt"):
        v = ratios[tier]
        if not v:
            continue
        vals = [r for r, _ in v]
        lo, lo_n = min(v)
        hi, hi_n = max(v)
        lines += [
            f"**{tier.upper()} vs ASC**, over {len(v)} variant"
            f"{'' if len(v) == 1 else 's'}: "
            f"{lo:.2f}x (`{lo_n}`) to {hi:.2f}x (`{hi_n}`), "
            f"mean {sum(vals) / len(vals):.2f}x.",
            "",
        ]

    faster = [n for r, n in ratios["pto"] if r < FASTER]
    if faster:
        lines += [
            "## PTO is faster than ASC on some variants",
            "",
            "Reported as measured. " + (
                f"{len(faster)} of {len(ratios['pto'])} variants come out below "
                f"{FASTER:.2f}x: " + ", ".join(f"`{n}`" for n in sorted(faster)) + "."),
            "",
            "The expected story going in was that ASC — the most verbose tier,",
            "naming the physical operation — would also be the fastest. On this",
            "sample that does not hold, and the measurement is what is reported.",
            "A static operation count is not a cycle count: where PTO uses a wider",
            "vector it also runs fewer loop iterations, which `tools/vf_lines.py`",
            "cannot see. All of these rows produce identical results, so the",
            "comparison is like-for-like.",
            "",
        ]

    if ratios["simt"]:
        hi, hi_n = max(ratios["simt"])
        lo, lo_n = min(ratios["simt"])
        vals = sorted(r for r, _ in ratios["simt"])
        rest = [r for r, n in ratios["simt"] if n != hi_n]
        # If that worst row ships an alternative spelling, quote its measured
        # effect rather than a remembered one.
        alt_ratio, alt_vs_asc = "fewer cycles", "lower"
        hw_ratio, hw_vs_asc = None, None
        base_id = f"simt/{hi_n}"
        for mode, _why in EXTRA.get(base_id, []):
            alt = records.get(f"{base_id}{EXTRA_SEP}{mode}", {})
            base = records.get(base_id, {})
            asc_c = records.get(f"asc/{hi_n}", {}).get("cycles")
            if mode == "hw_convert" and alt.get("cycles") and base.get("cycles"):
                # a different claim entirely -- not a respelling of the encoder
                # but its removal, so it is quoted separately below
                hw_ratio = base["cycles"] / alt["cycles"]
                hw_vs_asc = alt["cycles"] / asc_c if asc_c else None
                continue
            if alt.get("cycles") and base.get("cycles"):
                alt_ratio = (f"{base['cycles'] / alt['cycles']:.2f}x fewer "
                             f"cycles ({base['cycles']:,.0f} to "
                             f"{alt['cycles']:,.0f})")
                if asc_c:
                    alt_vs_asc = f"{alt['cycles'] / asc_c:.1f}x ASC"
        lines += [
            "## Reading the SIMT column",
            "",
            "SIMT is slower on every variant here, by between "
            f"{lo:.2f}x (`{lo_n}`) and {hi:.2f}x (`{hi_n}`). Three things are",
            "worth separating out.",
            "",
            f"**The top of that range is one variant, and it decomposes.**",
            f"`{hi_n}` at {hi:.0f}x is the FP4 output: no FP4 store is reachable",
            "from a `SimtVF` region, so an 18-line hand-written e2m1 encoder stands",
            "in for a hardware conversion the SIMD tiers simply name. Dropping that",
            f"one row, the range is {min(rest):.2f}x to {max(rest):.2f}x with a mean",
            f"of {sum(rest) / len(rest):.2f}x (median {vals[len(vals) // 2]:.2f}x).",
            "Read the headline mean with that in mind.",
            "",
            "That row was worth taking apart rather than filing as \"encoders are",
            "expensive\", and it splits into two independent factors:",
            "",
            "| factor | measured |",
            "|---|---|",
            "| more instructions than its sibling SIMT variants | 243,130 against 4,036-5,828 |",
            "| of those, pure control flow | 44,835 `SIMT_BRANCH` + 22,788 predicate sets |",
            "",
            "The second factor is **branch divergence**, and it is not inherent.",
            f"`{hi_n}` is the only variant in the ladder carrying a large number of",
            "`SIMT_BRANCH` and `SIMT_START_DVG` / `SIMT_END_DVG` markers, because",
            "`T.if_then_else` on a data-dependent per-element condition lowers to a",
            "real branch rather than a predicated select. That variant therefore",
            "**ships both spellings**, the harness asserts they are byte-identical,",
            "and both are measured -- see \"Alternative spellings\" below for the",
            "numbers and `harness/probe/simt_divergence.py` for the controlled",
            "experiment.",
            "",
            "The short version: the branch-free spelling issues 37% **fewer**",
            f"instructions and runs **{alt_ratio}** -- "
            f"{alt_vs_asc} rather than {hi:.1f}x -- because the instructions it",
            "drops are the branching itself.",
            "",
            "**And the encoder itself is optional.** No `T.Cast` to FP4 lowers",
            "inside a `SimtVF` region -- an fp4 element is four bits, the",
            "toolchain's only FP4 types are the vector ones, and a per-thread",
            "scalar has nothing to convert into",
            "(`harness/probe/simt_fp4_convert.py` records the nine spellings",
            "that fail). But the *kernel* can leave the region for that one",
            "instruction: a three-line `T.SimdVF` block in place of the",
            "eighteen-line encoder, same bytes, "
            + (f"**{hw_ratio:.1f}x fewer cycles**" if hw_ratio
               else "measured separately")
            + (f" and {hw_vs_asc:.2f}x ASC rather than {hi:.1f}x."
               if hw_vs_asc else "."),
            "That spelling ships in the same file behind `hw_convert=` and is",
            "in the table below. It is **not** what the SIMT column measures,",
            "because a kernel that drops into `T.SimdVF` is no longer answering",
            "what the thread model costs on its own.",
            "",
            "Two conclusions, and they pull in opposite directions, which is why",
            f"both are stated. The {hi:.2f}x is a true measurement of the kernel as",
            "written, and it is not a measurement artefact -- the trace audit",
            "confirms the right kernel with the right launch count. But that slice",
            "of it is a **codegen gap** rather than a cost of the thread model: on a",
            "GPU the conditional would be predicated into a select and cost nothing.",
            "The default keeps the natural spelling deliberately, since the",
            "comparison asked for was the one a GPU programmer writes first.",
            "",
            "**What SIMT gives up is thread occupancy, and the width of a",
            "`T.Parallel` is the whole of it.** This table measures that directly,",
            "because the SIMT tier was re-spelled once already. Every reduction in",
            "`per_token` and `per_block` first shipped as one thread per quant group",
            "(or per tile row) with an inner `T.serial`, which on `group_size = 32`",
            "is 4 busy threads out of 128. Written instead as one thread per element",
            "plus `T.reduce_absmax`, `per_token/01` went from 71,606 cycles to",
            "8,204 -- 11.59x ASC to 1.33x -- and `per_block/01` from 19,276 to",
            "2,664. `harness/probe/simt_reduce_max.py` measures the two spellings of",
            "one kernel side by side and checks they are byte-identical. Nothing",
            "about the arithmetic changed; scalar bit manipulation is essentially",
            "free either way.",
            "",
            "**Shorter source and fewer cycles are different axes.** Across all 22",
            "variants SIMT is 51% of ASC's source lines inside the VF region",
            "(`python tools/vf_lines.py`). The two axes used to separate most",
            "clearly on `per_token/05`; with the reduction spelled right that row",
            "is now a win on both (16 lines against ASC's 54, at 1.99x). Where they",
            "still part company is `per_channel/04`: 17 lines against ASC's 46, and",
            "8.08x the cycles. The thread model removes the lane-placement problem",
            "there -- an interleaved byte pack that costs ASC a `vintlv` and a",
            "padded scratch row -- and leaves most of the vector unit idle doing it.",
            "",
            "**The reductions are spelled for speed; the format work is not.**",
            "After the re-spelling above, what remains of the SIMT gap is where a",
            "SIMD frontend *names* an instruction that the thread model has to",
            "write out: the e2m1 encoder (`per_token/04`, `per_block/03`), the",
            "UE8M0 byte pack (`per_channel/02`, `/04`) and the elementwise",
            "`cast_back` family, which has no reduction to win and sits at",
            "3.20-4.42x on scalar issue alone. None of those were tuned, and the",
            "next section says how far tuning moves such a kernel.",
            "",
            "## Do not generalise this table past this sample",
            "",
            "Every row here has SIMT slower, and it would be easy to read that as",
            "a law. It is not one, for two reasons.",
            "",
            "**These are 22 single-config quantization kernels, one shape each.**",
            "The reductions all have `group_size = 32`. That was the case that",
            "left the most threads idle until the fold was handed to",
            "`T.reduce_absmax`; a kernel whose natural parallelism matches the",
            "thread count differently would not look like this either way.",
            "",
            "**Tuning moves it by more than the gap on some variants.** A separate",
            "SIMT-versus-SIMD suite, measured on the same simulator and SoC but",
            "not part of this repo, swept several SIMT strategies per kernel and",
            "found ratios from **0.33x to 53x** — that is, SIMT *winning* by 3x on",
            "some grouped reductions and losing by 50x elsewhere. In one case the",
            "strategy choice alone (keeping values in registers versus streaming",
            "through UB) moved the ratio from 0.63x to 0.33x, a factor of two, on",
            "an identical kernel.",
            "",
            "So the honest scope of this table is: *for these kernels, at this",
            "shape, written the obvious way, SIMT costs this much more.* It",
            "supports the claim that the thread model trades cycles for ease. It",
            "does not support a claim about SIMT in general, and the four-tier",
            "story in `tiers/simt.md` is written not to make one.",
            "",
        ]

    # Derive the call_count audit rather than asserting it: the method doc's
    # guard is only worth anything if it is actually applied to every row.
    mismatch = []
    for name in sorted(by_variant):
        row = by_variant[name]
        counts = {t: row[t].get("call_count") for t in ("asc", "pto", "simt")
                  if t in row and row[t].get("result") == "PASS"
                  and row[t].get("call_count")}
        if len(set(counts.values())) > 1:
            mismatch.append((name, counts))

    lines += [
        "## Checking a comparison is like-for-like",
        "",
        "`call_count` is the guard the method doc insists on: *a large speed-up",
        "with a mismatched `call_count` is a different kernel, not a faster one.*",
        "Every row above was checked against it.",
        "",
    ]
    if not mismatch:
        lines += [
            f"**All {len(ratios['simt']) or len(by_variant)} comparable variants "
            f"issue the same number of VF launches in every tier**, so no ratio "
            f"here is explained by one tier simply doing less work.",
            "",
        ]
    else:
        lines += [
            f"{len(mismatch)} variant(s) differ, and each needs a reason before "
            f"its ratio means anything:",
            "",
            "| variant | ASC | PTO | SIMT |",
            "|---|---:|---:|---:|",
        ]
        for name, counts in mismatch:
            lines.append(f"| `{name}` | " + " | ".join(
                str(counts.get(t, "—")) for t in ("asc", "pto", "simt")) + " |")
        lines += [
            "",
            "A SIMT kernel issuing *fewer* launches than the SIMD tiers is the",
            "expected direction where the SIMD tiers set up index vectors that",
            "the thread model does not need — a transposed store is the clearest",
            "case. A SIMT kernel issuing more would want explaining.",
            "",
        ]
    driven = sorted(f"`{'/'.join(v.split('/')[1:])}`" for v in records
                    if records[v].get("method") == "driver")
    if driven:
        uniq = sorted(set(driven))
        lines += [
            "## Two variants are measured differently, and why",
            "",
            f"{', '.join(uniq)} are measured through `tools/vf_drive.py` rather",
            "than through `python -m harness.check`, at `mode=\"full\"`.",
            "",
            "`msprof op simulator` profiles a **single operator** -- in practice the",
            "first one the process launches. Both of these variants take an input",
            "scale tensor that their unused modes never read, and their `launch()`",
            "allocates a placeholder for it with `torch.zeros(..., device=...)`,",
            "which on the NPU is a fill *kernel*. That fill ran first, so the",
            "ordinary path measured it instead of the variant.",
            "",
            "It was caught because all three tiers reported **exactly the same**",
            "cycle count, and three different implementations cannot cost the same.",
            "`tools/vf_cycles.py` now checks for that tie on every variant, so this",
            "particular mistake cannot come back silently.",
            "",
            "The driver builds every tensor on the host, moves them with a plain",
            "copy, and calls `compile_kernel` directly once, which makes the",
            "variant's own kernel the first operator. Two independent signs that",
            "the replacement numbers are right: the launch counts now match the",
            "other tiers', and for PTO and SIMT the `per_token/06` figures come out",
            "*identical* to `per_token/01` -- which is what trace-time mode",
            "selection should produce, since `full` mode emits variant 01's kernel.",
            "ASC's does not, and does not claim to: its four-mode body is",
            "restructured (3 barriers against 2, 11 vector loads against 9), and it",
            "costs about 11% more.",
            "",
        ]
    alts = [(base, mode, why, records[f"{base}{EXTRA_SEP}{mode}"])
            for base, lst in EXTRA.items() for mode, why in lst
            if f"{base}{EXTRA_SEP}{mode}" in records]
    if alts:
        lines += [
            "## Alternative spellings, shipped and measured",
            "",
            "These are not variants. Each is a second spelling of one variant's",
            "kernel, shipped in the same file behind a trace-time flag, verified",
            "by the harness to be byte-identical to the default, and measured",
            "here so the cost of the choice is on the record rather than implied.",
            "They are kept out of the table above, which stays one row per",
            "variant.",
            "",
            "| variant | spelling | cycles | speed-up | vs ASC |",
            "|---|---|---:|---:|---:|",
        ]
        f = lambda v: "—" if v is None else f"{v:,.0f}"
        # one default row per variant, however many alternatives it has
        seen: set[str] = set()
        for base, mode, why, rec in alts:
            tier, kernel, stem = base.split("/")
            asc = records.get(f"asc/{kernel}/{stem}", {}).get("cycles")
            d = records.get(base, {}).get("cycles")
            c = rec.get("cycles")
            if base not in seen:
                seen.add(base)
                lines.append(
                    f"| `{kernel}/{stem}` | the shipped default | {f(d)} | "
                    f"1.00x | {f'{d / asc:.1f}x' if d and asc else '—'} |")
            lines.append(
                f"| `{kernel}/{stem}` | `{mode}` | **{f(c)}** | "
                f"**{f'{d / c:.2f}x' if c and d else '—'}** | "
                f"**{f'{c / asc:.2f}x' if c and asc else '—'}** |")
        lines += [
            "",
            "What each one is:",
            "",
        ]
        for _base, mode, why, _rec in alts:
            lines.append(f"* **`{mode}`** -- {why}.")
        lines += [
            "",
            "`branch_free` issues **fewer** instructions than the default and",
            "runs in fewer cycles, because `T.if_then_else` on a data-dependent",
            "per-element condition lowers to a real branch on this toolchain",
            "while the mask form lowers to a select; the instructions it drops",
            "are the branching. `hw_convert` is not a respelling of the encoder",
            "but its removal, and it is the larger effect by far.",
            "",
            "The default keeps the hand-written conditional encoder",
            "deliberately: the SIMT column measures what the thread model costs",
            "on its own, and both alternatives change that question --",
            "`branch_free` by writing what a GPU compiler would have emitted,",
            "`hw_convert` by leaving the region. All three are verified",
            "byte-identical by the harness. `harness/probe/simt_divergence.py`",
            "and `harness/probe/simt_fp4_convert.py` are the controlled",
            "experiments, and `doc/known-issues.md` records both.",
            "",
        ]
    lines += [
        "## Confirming the trace is the quant kernel",
        "",
        "A cycle total cannot say *what* it profiled: the CSV's VF row is one",
        "aggregated launch. The simulator's instruction log can, because it names",
        "every instruction that issued, vector pipes included. "
        "`python tools/vf_trace_audit.py` reads that log for every measured id and",
        "checks the mnemonics are the ones a quantization kernel is made of.",
        "",
        "All 66 traces pass. The instruction mix matches the source closely enough",
        "to be checked by hand -- `asc/cast_back/01`, at K=128 with 64-lane",
        "vectors, is two strips per token, and its log shows **per launch**:",
        "",
        "| mnemonic | per launch | the source line |",
        "|---|---:|---|",
        "| `RV_VLDI` | 6 | 1 FP8 load + 2 `BRC_B32` scale loads, x2 strips |",
        "| `RV_VCVT_F2F` | 4 | the in and out converts, x2 strips |",
        "| `RV_VSEL` | 2 | `S.vsel` picking between two group scales |",
        "| `RV_VMUL` | 2 | `S.vmul` applying the scale |",
        "| `RV_VSTI` | 2 | `S.vsts` |",
        "",
        "Every count is the source's per-strip structure times two strips. The",
        "SIMT tier shows the same thing in its own instruction set (`SIMT_F2F`,",
        "`SIMT_FMUL`, `SIMT_FMNMX`, `SIMT_LDS`), and `simt/per_token/04` shows it",
        "most convincingly by having **zero** converts: that variant takes float32",
        "input and writes hand-packed FP4 nibbles, so its format work is 66,138",
        "integer shift and logic instructions instead -- which is both what its",
        "source predicts and why it costs 99.62x.",
        "",
        "The negative control is reproducible. Pointed at the `torch.zeros` fill",
        "that was measured by mistake, the same audit reports one `RV_VDUPS`, two",
        "stores, and no multiply at all, and exits non-zero:",
        "",
        "```bash",
        "python tools/vf_trace_audit.py --dump <OPPROF_dir>",
        "```",
        "",
        "Per-variant commentary on *why* each row looks the way it does is on the",
        "variant pages under `doc/quant/`; [`tiers/simt.md`](tiers/simt.md)",
        "summarises where the thread model wins and where it does not.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {DATA.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="variant ids or selectors")
    ap.add_argument("--all", action="store_true", help="every vector-tier variant")
    ap.add_argument("--table", action="store_true", help="print the cached table only")
    ap.add_argument("--drive", action="store_true",
                    help="re-measure the DRIVEN variants through tools/vf_drive.py")
    ap.add_argument("--report", action="store_true",
                    help="write perf/quant_vf_cycles.json and the markdown summary")
    args = ap.parse_args()

    records: dict = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    if args.table:
        print(build_table(records))
        if args.report:
            write_report(records)
        return 0

    if args.drive:
        todo = [(vid, DRIVEN["/".join(vid.split("/")[1:])], vid)
                for vid in records
                if EXTRA_SEP not in vid
                and "/".join(vid.split("/")[1:]) in DRIVEN]
        todo += [(base, mode, f"{base}{EXTRA_SEP}{mode}")
                 for base, alts in EXTRA.items() if base in records
                 for mode, _ in alts]
        if not todo:
            print("nothing to drive; run --all first so the harness verdicts exist")
            return 1
        print(f"re-measuring {len(todo)} kernel(s) through tools/vf_drive.py\n")
        for base, mode, key in sorted(todo, key=lambda t: t[2]):
            rec = measure_driven(base, mode, records.get(base))
            rec["id"] = key
            records[key] = rec
            vid = key
            cyc = "-" if rec.get("cycles") is None else f"{rec['cycles']:,.0f}"
            print(f"  {vid:34} {str(rec['result']):>8} {cyc:>10} cycles  "
                  f"calls={rec.get('call_count', 0)}  mode={mode}")
            CACHE.write_text(json.dumps(records, indent=1, sort_keys=True))
        print()
        print(build_table(records))
        if args.report:
            write_report(records)
        return 0

    sys.path.insert(0, str(ROOT))
    from harness.check import resolve                      # noqa: E402

    if args.all:
        pairs = [(v, t) for v, t in resolve([]) if t in VECTOR_TIERS]
    elif args.query:
        pairs = [(v, t) for v, t in resolve(args.query) if t in VECTOR_TIERS]
    else:
        ap.error("give one or more selectors, or --all")

    if not pairs:
        print("no vector-tier variants matched (torch has no simulator run)")
        return 1

    print(f"measuring {len(pairs)} variant(s) under the simulator\n")
    for variant, tier in pairs:
        vid = variant.id(tier)
        rec = measure(vid, effective_shape(variant, tier))
        records[vid] = rec
        cyc = "-" if rec.get("cycles") is None else f"{rec['cycles']:,.0f}"
        extra = f"  {rec['note']}" if rec.get("note") else ""
        print(f"  {vid:34} {rec['result']:>8} {rec.get('instr') or '-':>8} "
              f"{cyc:>10} cycles  calls={rec.get('call_count', 0)}{extra}")
        CACHE.write_text(json.dumps(records, indent=1, sort_keys=True))

    print()
    print(build_table(records))
    if args.report:
        write_report(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
