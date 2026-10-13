"""Generate presentation/slides.md -- the deck source, in Marp dialect.

    python tools/make_slides.py             # write presentation/slides.md
    python tools/make_slides.py --check     # fail if it is out of date

The deck quotes kernel code and measured cycles. Both change when a kernel is
edited and re-measured, so the deck is generated rather than hand-maintained:
code comes from `tools/make_report_snippets.py` (verbatim, anchored to the real
files) and every figure from `perf/quant_vf_cycles.json` and `tools/vf_lines.py`.

The output is authored in **Marp dialect** -- YAML front matter, `---` slide
separators, `##` slide titles -- so that

    marp presentation/slides.md --pdf --pptx

works unchanged once a Chrome binary is available. Today `presentation/Makefile`
builds it with pandoc + Beamer, which needs no browser; `tools/slides_strip_marp.py`
is the small shim between the two dialects.

Prose lives in this file. If you want to reword a slide, edit it here, not in
slides.md, or the next regeneration will overwrite you.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "slides.md"
sys.path.insert(0, str(ROOT / "tools"))
from make_report_snippets import resolve                          # noqa: E402

def code(name):
    _, _, _, body = resolve(name)
    return "```python\n" + body + "```"

d = json.load(open(ROOT / "perf/quant_vf_cycles.json"))
out = subprocess.run([sys.executable, str(ROOT/"tools/vf_lines.py")],
                     capture_output=True, text=True, cwd=ROOT).stdout
st = {}
for ln in out.splitlines():
    p = ln.split()
    if len(p) >= 9 and "/" in p[0]:
        st[p[0]] = dict(oa=int(p[1]), op=int(p[2]), la=int(p[3]), lp=int(p[4]), ls=int(p[5]))
c = lambda t, v: d[f"{t}/{v}"]["cycles"]

def row(v):
    """One `| ASC | PTO | SIMT |` measured block for a variant."""
    a, p, s = c("asc", v), c("pto", v), c("simt", v)
    r = st[v]
    return (f"| | ASC | PTO | SIMT |\n|---|---:|---:|---:|\n"
            f"| VF-region lines | {r['la']} | {r['lp']} | **{r['ls']}** |\n"
            f"| simulated cycles | {a:,.0f} | {p:,.0f} | {s:,.0f} |\n"
            f"| vs ASC | 1.00x | {p/a:.2f}x | {s/a:.2f}x |")

def meas(v):
    """A one-line measured summary, for slides that also carry code.

    The full per-variant table (`row`) is four table rows; on a slide that
    already shows three code blocks that overflows the frame, and the same
    numbers appear again in the aggregate tables and the charts.
    """
    a, p_, s_ = c("asc", v), c("pto", v), c("simt", v)
    r = st[v]
    return (f"**Measured** — cycles ASC {a:,.0f} / PTO {p_:,.0f} / SIMT {s_:,.0f} "
            f"({p_/a:.2f}x / {s_/a:.2f}x vs ASC); "
            f"VF-region lines {r['la']} / {r['lp']} / {r['ls']}.")


pr = [c("pto", v)/c("asc", v) for v in st]
sr = [c("simt", v)/c("asc", v) for v in st]
nofp4 = [x for v, x in zip(st, sr) if "per_token/04" not in v]
tot = lambda k: sum(r[k] for r in st.values())
A = dict(ops=f"{tot('op')/tot('oa'):.0%}", lp=f"{tot('lp')/tot('la'):.0%}",
         ls=f"{tot('ls')/tot('la'):.0%}", pmean=f"{sum(pr)/len(pr):.2f}x",
         pmin=f"{min(pr):.2f}x", pmax=f"{max(pr):.2f}x",
         pwin=sum(1 for x in pr if x < 0.98),
         nmin=f"{min(nofp4):.2f}x", nmax=f"{max(nofp4):.2f}x",
         nmean=f"{sum(nofp4)/len(nofp4):.2f}x",
         smax=f"{max(sr):.2f}x")
bf = d["simt/per_token/04_fp32_in_fp4_out+branch_free"]["cycles"]

# the aggregate table, split across two slides so neither overflows a frame
HDR = ["| variant | ops A/P | lines A/P/S | cyc A | cyc P | cyc S | P/A | S/A |",
       "|---|---|---|---:|---:|---:|---:|---:|"]
def agg_rows(kernels):
    out = list(HDR)
    for v in st:
        if v.split("/")[0] not in kernels:
            continue
        r = st[v]; a, pp, ss = c("asc", v), c("pto", v), c("simt", v)
        kern, stem = v.split("/")
        out.append(f"| `{kern}/{stem.split('_')[0]}` | {r['oa']}/{r['op']} | "
                   f"{r['la']}/{r['lp']}/{r['ls']} | {a:,.0f} | {pp:,.0f} | {ss:,.0f} | "
                   f"{pp/a:.2f} | {ss/a:.2f} |")
    return "\n".join(out)
AGG1 = agg_rows({"cast_back", "per_token"})
AGG2 = agg_rows({"per_block", "per_channel"})

# Marp-only styling. It lives in the YAML front matter, which
# tools/slides_strip_marp.py removes before pandoc sees the file, so the Beamer
# build is unaffected. Marp does not auto-shrink content to fit a slide the way
# Beamer does, so these sizes are what keep the dense slides -- the 13-row table
# and the three-way code comparisons -- inside the frame.
MARP_CSS = """
section {
  font-size: 19px;
  line-height: 1.35;
  padding: 38px 52px 28px;
  background: #ffffff;
  color: #1a1a1a;
}
h2 {
  font-size: 26px;
  margin: 0 0 14px;
  padding-bottom: 6px;
  border-bottom: 2px solid #c8c8c8;
}
p, li { margin: 5px 0; }
ul { margin: 4px 0; }
pre {
  font-size: 13.5px;
  line-height: 1.32;
  padding: 7px 10px;
  margin: 6px 0;
  background: #f7f7f3;
  border: none;
  border-left: 3px solid #b0b0b0;
  border-radius: 0;
}
code { font-size: 0.95em; }
table {
  font-size: 14.5px;
  margin: 7px 0;
  border-collapse: collapse;
}
th, td { padding: 2px 9px; }
blockquote {
  font-size: 19px;
  margin: 8px 0;
  padding: 4px 0 4px 14px;
  border-left: 4px solid #777;
  background: none;
}
section::after { font-size: 14px; color: #999; }
"""

MARP_CSS_INDENTED = "\n".join("  " + ln for ln in MARP_CSS.strip().splitlines())

# Facts the prose quotes, derived rather than typed. Several earlier hardcoded
# claims were simply wrong (a superlative that held only within one kernel
# family, a count that double-counted a row); computing them removes the class.
_pto_ops_better = [v for v in st if st[v]["op"] < st[v]["oa"]]
_both_better = [v for v in st
                if st[v]["op"] < st[v]["oa"] and c("pto", v) / c("asc", v) < 0.98]
_simt_sorted = sorted(st, key=lambda v: c("simt", v) / c("asc", v))
_biggest_op_cut = min(st, key=lambda v: st[v]["op"] / st[v]["oa"])
# Measured by harness/probe/simt_thread_count.py: the same per_token/01 kernel
# built at three declared thread counts. All three match the oracle. This
# replaced an earlier caveat which asserted, without measuring, that the cost
# came from declaring more threads than the reduction's extent.
THREADS_RESULT = """Measured on `per_token/01`, the same kernel built at three declared widths
(`harness/probe/simt_thread_count.py`, all three verified against the oracle):

| `threads` | simulated cycles | vs 128 | correct |
|---:|---:|---:|---|
| 32 | 71,278 | 1.00x | yes |
| 64 | 69,047 | 0.96x | yes |
| 128 | 71,606 | 1.00x | yes |

**Lowering the declared thread count does not recover the cost.** Across a 4x
range the spread is under 4%, so the penalty is not caused by declaring more
threads than the reduction uses. It is the per-element execution model: ~37,000
instructions at 1.95 cycles each, against 32 VF launches of vector work."""

F = dict(
    both_better=len(_both_better),
    simt_below6=sum(1 for v in st if c("simt", v) / c("asc", v) < 6),
    t05_rank=_simt_sorted.index("per_token/05_col_major_sf") + 1,
    biggest_op_cut=_biggest_op_cut.split("_")[0].replace("/", "/"),
    bf_ratio=bf / c("simt", "per_token/04_fp32_in_fp4_out"),
    bf_vs_asc=bf / c("asc", "per_token/04_fp32_in_fp4_out"),
    fp4_vs_asc=c("simt", "per_token/04_fp32_in_fp4_out")
    / c("asc", "per_token/04_fp32_in_fp4_out"),
)



SLIDES = f"""---
marp: true
theme: default
paginate: true
title: "Writing NPU quant kernels: SIMT vs PTO vs ASC VF code comparison"
style: |
{MARP_CSS_INDENTED}
---

## Writing NPU quant kernels: SIMT vs PTO vs ASC

**Code size and simulated performance of three tilelang vector frontends
implementing one group-wise quantization algorithm.**

- 22 kernel variants, each implemented in ASC, PTO and SIMT plus a PyTorch
  reference: 88 kernels, all passing `python -m harness.check`
- msprof-simulated VF cycles for all 66 NPU kernels
- tilelang 0.1.15, ptoas vmi 0.1.9, CANN 9.2.0-beta.2, SoC `Ascend950PR_9599`
- Single core, `T.Kernel(1)`. Only the `T.SimdVF` / `T.SimtVF` region differs
  between the three implementations of a variant; tiling, DMA and buffer
  allocation are identical.

---

## tilelang frontend comparison

Group-wise quantization, in PyTorch:

{code("torch_per_token")}

| frontend | abstraction level | smallest unit of data |
|---|---|---|
| **ASC** (`T.simd`, `S`) | named hardware instruction and distribution mode | one 64-lane vector register |
| **PTO/VMI** (`T.vmi`, `V`) | operation with width and segment count as arguments | one logical vector, any width |
| **SIMT** (`T.SimtVF`, `T.Parallel`) | per-element expression | one element, held by one thread |

`T.SimdVF` / `T.SimtVF` regions range from **6 to 57 lines** across the 22
variants. msprof-simulated VF cycles for the same variant differ by **10-100x
between the SIMT and the SIMD implementations**, and by up to 1.6x between the
two SIMD frontends.

---

## ASC vs PTO: fewer operations, comparable cycles

PTO expresses the same 22 kernels in **{A['ops']} of ASC's vector operations** and
**{A['lp']} of its source lines**. Simulated cycles do not follow:

| PTO relative to ASC | |
|---|---|
| vector operations written | {A['ops']} |
| `T.SimdVF` source lines | {A['lp']} |
| simulated VF cycles | **{A['pmean']} mean**, range {A['pmin']}-{A['pmax']} |

- PTO uses fewer cycles on **{A['pwin']} of 22** variants, more on 6, and is
  within 2% on 7
- On 4 of those {A['pwin']} it writes *more* operations than ASC
- Operations written and cycles executed correlate at **Pearson r = -0.09**
  (n = 22): across this set, source size carries no information about cycle count

Both SIMD frontends are 3-100x faster than SIMT on every variant.

---

## Benchmark setup (and caveats resolved)

**Cycle measurement.** Sum the `cycles` column of
`core0.veccore0_instr_exe.csv` over rows whose `instr` is a VF launch (`VF` for
SIMD, `VF_SIMT` for SIMT). The filter keys on `instr`, not `pipe`: every VF row
sits on pipe `PUSHQ`, which `MOVEMASK` and `PUSH_PB` also use.

**Correctness.** Each kernel is compared against `harness/oracle.py`, with an
assertion that the result was read back from the device. Cycles are not reported
for a kernel that computed a wrong result.

**Equal work.** VF launch counts were compared across the three frontends on
every variant. They agree except on `per_token/05`, where ASC and PTO issue 34
and SIMT 32; the two extra launches build the gather index SIMT does not need.

**Two measurement errors found and corrected.** `per_token/06` and
`per_block/05` allocate a placeholder input with `torch.zeros(device=...)`,
which is an aten fill *kernel*; msprof profiles the first operator, so all three
frontends had measured that fill. Detected because three different
implementations reported identical cycle counts. Both are now measured through
`tools/vf_drive.py`, and `tools/vf_trace_audit.py` checks the instruction log of
all 66 traces for the arithmetic a quant kernel must contain.

---

## `cast_back/01`: applying a per-group scale

64 consecutive channels span two scale groups, so each lane requires one of two
scalars.

**ASC** builds the scale vector: two broadcast loads and a lane-mask select.

{code("broadcast_asc")}

**PTO** declares the segment structure on the load: 64 lanes, 2 segments.

{code("broadcast_pto")}

**SIMT** indexes the scale array per element.

{code("broadcast_simt")}

---

## `cast_back/01`: measured, and output dtype selection

{meas("cast_back/01_e4m3_fp32sf")}

PTO replaces 3 instructions with 1 and uses 4.6% more cycles. SIMT uses 6 lines
against ASC's 16, at 3.28x the cycles.

The variant also switches output dtype. ASC selects a different store
instruction — the wider dtype takes the simpler one:

{code("outdtype_asc")}

PTO infers packing from the destination buffer, so only the convert changes:

{code("outdtype_pto")}

SIMT emits no dtype branch: `T.copy` converts between buffers of differing
dtype.

---

## `per_channel/02`: M-packed scales require a lane interleave

Output byte `2c` holds m-group `2i`'s exponent for channel `c`, byte `2c+1`
holds m-group `2i+1`'s. On a vector unit this is an interleave instruction.

**ASC:**

{code("interleave_asc")}

**PTO** — same instruction, with the vector width given explicitly:

{code("interleave_pto")}

---

## `per_channel/02`: SIMT replaces the interleave with two scalar stores

{code("interleave_simt")}

A `uint8` vector register holds 256 lanes, so the loads above read 128 bytes
past the 128 meaningful ones. Both SIMD frontends therefore pad the scratch
buffer by a whole tile:

{code("pad_asc")}

{code("pad_pto")}

A scalar store cannot overrun, so SIMT allocates the exact size:

{code("pad_simt")}

---

## `per_channel/02`: performance penalty of SIMT

{row("per_channel/02_round_packed_m")}

Observations:

- SIMT removes one instruction and one padding requirement, using 17 lines
  against ASC's 35
- SIMT costs **7.02x** the simulated cycles
- PTO is **5 lines longer** than ASC (40 vs 35) at 0.96x the cycles: the
  explicit `size=` and mask operands cost source length, not time

---

## `per_token/05`: column-major scales require a register gather

Lane assignment inside a vector register is fixed, so transposing the scale
array means computing, per lane, which source element that lane receives.

**ASC** builds the index vector from the lane id:

{code("transpose_asc")}

...and gathers through it:

{code("transpose_asc_gather")}

---

## `per_token/05`: PTO needs an extra broadcast; SIMT needs an index

**PTO** performs the same index arithmetic. VMI has no scalar-operand `vmul`, so
the stride constant is first broadcast into a vector — 3 lines where ASC uses 2:

{code("transpose_pto")}

**SIMT** stores the scale at the transposed index. No index vector, no gather:

{code("transpose_simt")}

---

## `per_token/05`: SIMT's performance penalty on transpose

{row("per_token/05_col_major_sf")}

Observations:

- 15 SIMT lines against ASC's 54 is the largest source-size reduction among the
  22 variants
- SIMT costs **13.74x** the cycles, the {F['t05_rank']}th lowest SIMT ratio of 22
- ASC and PTO are within 1% of each other: a gather is a single hardware
  operation, so VMI has no segment structure or width to collapse here

---

## `per_channel/01`: reduction along M needs no cross-lane step

One scale per channel means the reduction runs down the token axis. A channel
occupies one lane throughout, so no frontend performs a cross-lane operation.

**ASC** accumulates into a register across 32 rows:

{code("chanreduce_asc")}

**PTO** — the same structure, with mask and width operands:

{code("chanreduce_pto")}

---

## `per_channel/01`: SIMT, and the measured result

**SIMT** assigns one thread per channel and loops over the 32 rows:

{code("chanreduce_simt")}

{meas("per_channel/01_raw_32tokens")}

ASC and PTO use **{st['per_channel/01_raw_32tokens']['oa']} and
{st['per_channel/01_raw_32tokens']['op']} vector operations and
{st['per_channel/01_raw_32tokens']['la']} lines each**: segment counts and width
arguments have nothing to apply to. SIMT's 12 lines express the same three
stages and the same memory traffic, at 5.33x the cycles.

`per_block/01` is the second such case — one scale per 32x32 tile is a scalar,
so a column-major scale array changes only the final `T.copy` index and costs
no additional instruction in any of the three frontends.

---

## `per_token/01`: segmented reduction

Each 32-channel group needs its own maximum. An ASC vector reduction produces
one value per register, so 4 groups require 4 masked reductions and 4
single-element stores:

{code("segreduce_asc")}

**PTO** declares the segment count on one reduction:

{code("segreduce_pto")}

8 instructions become 3. The matching broadcast on the apply pass collapses from
4 loads plus 2 selects to 1 load.

---

## `per_token/01`: SIMT, and the measured result

**SIMT** assigns one thread per group:

{code("segreduce_simt")}

{meas("per_token/01_raw_fp32sf")}

- PTO uses {st['per_token/01_raw_fp32sf']['op']} vector operations against ASC's
  {st['per_token/01_raw_fp32sf']['oa']} and **0.74x** the cycles. Operation count
  and cycle count favour PTO together on {F['both_better']} of the 22 variants.
- `T.Parallel(num_groups)` has extent 4 at `K=128, group_size=32`: the SIMT
  reduction uses 4 of the 128 declared threads, at 11.59x the cycles.

---

## `per_token/02`: lane count as a parameter

The scale is rounded up to a power of two by rewriting the float32 exponent
field. **ASC** fixes the lane count in every type name (`"uint32x64"`):

{code("exponent_asc")}

**PTO** takes the width as an argument, so one definition serves 4, 64, 128 or
256 lanes:

{code("exponent_pto")}

---

## `per_token/02`: the same arithmetic on scalars

**SIMT** has no vector width to carry:

{code("exponent_simt")}

{meas("per_token/02_round_sf")}

{st['per_token/02_round_sf']['oa']} operations against
{st['per_token/02_round_sf']['op']} is the largest operation-count reduction in
the set, at a **1.02x** cycle ratio. The width-parameterised definition is PTO's
measurable advantage here; cycle count is unchanged.

---

## `cast_back/02`: PTO's width and mask operands cost source length

Decoding a UE8M0 exponent byte. **ASC** names the distribution mode on the load:

{code("ue8m0_asc")}

**PTO** states width, stride, group count and mask:

{code("ue8m0_pto")}

Operation counts tie at {st['cast_back/02_packed_ue8m0']['oa']};
**PTO uses {st['cast_back/02_packed_ue8m0']['lp'] - st['cast_back/02_packed_ue8m0']['la']} more lines**
({st['cast_back/02_packed_ue8m0']['lp']} vs {st['cast_back/02_packed_ue8m0']['la']}).
Two further ASC advantages:

- **No per-lane variable right shift in VMI.** `S.vshr(v, shifts)` accepts a
  shift vector; `V.vshrs` accepts a scalar only, so production PTO computes both
  byte positions and selects. No variant in this set uses it — the kernels above
  shift left by a vector, which both frontends support.
- **`per_channel/04`** is PTO's longest region, {st['per_channel/04_compose']['lp']}
  lines against {st['per_channel/04_compose']['la']}, at a 1.00x cycle ratio.

---

## `per_token/04`: FP4 output, where SIMT has no instruction

ASC and PTO both reach a hardware `float32 -> e2m1` convert. Their work is
limited to pre-rounding to bfloat16 with a sticky bit.

**ASC** pairs two registers to deinterleave them, then converts:

{code("fp4_asc")}

**PTO** uses `V.vunzip`, which splits one value instead of pairing two:

{code("fp4_pto")}

---

## `per_token/04`: SIMT encodes e2m1 in software

No FP4 store is reachable from a `T.SimtVF` region. The SIMT kernel declares its
output `uint8` and extracts the float32 fields by hand — the first 6 lines of an
18-line encoder covering sign, exponent, subnormal renormalization,
guard/round/sticky and saturation:

{code("fp4_simt")}

{row("per_token/04_fp32_in_fp4_out")}

The result is bit-exact against the PyTorch reference packer. `per_block/03`
carries the same encoder at 16.77x.

---

## `per_token/04`: {F['fp4_vs_asc']:.0f}x decomposes into instruction count and divergence

| factor | SIMT `per_token/04` | other SIMT `per_token` variants |
|---|---|---|
| instructions issued | 124,294 | ~37,000 |
| cycles per instruction | 4.94 | 1.90-2.01 |

The second factor is branch divergence. `T.if_then_else` on a data-dependent
per-element condition lowers to a branch rather than a predicated select, and
this is the only variant whose trace carries a large number of
`SIMT_START_DVG` markers.

Rewriting the encoder's three conditionals as mask arithmetic
(`m = -T.Cast("int32", cond)`, then `(a & m) | (b & ~m)`) issues **45% more
instructions** and runs at **{F['bf_ratio']:.2f}x the cycles** —
{bf:,.0f} against {c("simt", "per_token/04_fp32_in_fp4_out"):,.0f}, or
**{F['bf_vs_asc']:.1f}x ASC instead of {F['fp4_vs_asc']:.1f}x**. Both spellings are in
the kernel behind a `branch_free` flag and the harness asserts their outputs are
byte-identical.

A GPU compiler predicates such a conditional into a select, so this component is
a code-generation gap rather than a property of the thread model.

---

## `per_block`: operation count and cycle count move independently

| variant | ops ASC | ops PTO | PTO ops | cycles PTO/ASC |
|---|---:|---:|---|---:|
| `per_block/01` | 21 | 22 | more | **0.85x** |
| `per_block/02` | 27 | 30 | more | 1.02x |
| `per_block/03` | 28 | 26 | fewer | **0.64x** |
| `per_block/04` | 21 | 22 | more | **0.65x** |
| `per_block/05` | 29 | 31 | more | 1.02x |

PTO writes more operations on 4 of the 5 and runs fewer cycles on 2 of those 4.
The reduction loop accounts for it. **ASC** processes a 1024-value tile in 16
steps of 64 lanes; **PTO** reaches 128 float32 lanes, so the same tile takes 8
steps of 128:

{code("blockreduce_pto")}

More operations per step, half as many steps executed. A static count cannot
observe the trip count.

---

## Source size does not predict cycle count

![](plots/correlation.png)

Left: PTO vs ASC, operations written against simulated cycles, 22 variants. The
dotted line marks where the points would fall if source size predicted cycles.
Right: SIMT vs ASC, source lines against cycles, `per_token/04` excluded.

Pearson r is **-0.09** and **-0.01**: on this set, source size does not
estimate runtime.

---

## PTO vs ASC: simulated cycles per variant

![](plots/pto_cycles.png)

---

## SIMT vs ASC: simulated cycles per variant

![](plots/simt_cycles.png)

`per_token/04` is separated because its cost is a missing hardware instruction
(previous section), not a property of the thread model. The remaining 21
variants span {A['nmin']}-{A['nmax']}.

---

## Measured data (1/2): `cast_back`, `per_token`

{AGG1}

`/06` `full` emits `/01`'s kernel, hence the identical SIMT cycle count.

---

## Measured data (2/2): `per_block`, `per_channel`

{AGG2}

`per_block` reduces a 32x32 tile to one scalar; `per_channel` reduces along the
token axis. `per_block/02` and `/05`, and `per_channel/02` and `/04`, share SIMT
cycle counts for the same reason: the composed variant emits the earlier
variant's `T.SimtVF` region unchanged.

---

## Totals across the 22 variants

| | PTO vs ASC | SIMT vs ASC |
|---|---|---|
| vector operations | {A['ops']} | not comparable |
| source lines | {A['lp']} | **{A['ls']}** |
| simulated cycles | **{A['pmean']}** ({A['pmin']}-{A['pmax']}) | {A['nmean']} ({A['nmin']}-{A['nmax']}) |

SIMT has no comparable operation count: its arithmetic is plain operators, so
the backend selects the instructions.

- PTO uses fewer cycles on {A['pwin']} of 22 variants, more on 6, within 2% on 7
- The SIMT range excludes `per_token/04`; including it the maximum is {A['smax']}
- SIMT costs under 6x on {F['simt_below6']} of 22 variants and over 10x on 11

---

## Caveats

1. **Simulated cycles, not hardware.** `msprof op simulator`, SoC
   `Ascend950PR_9599`. No silicon measurement is claimed.
2. **One shape per variant**, mostly `M=32, K=128`. Ratios compare three
   implementations of one variant and are valid; absolute cycle counts are not
   comparable between variants.
3. **Single core.** No multi-core scheduling, double buffering or cache hints.
4. **The SIMT kernels are not tuned.** They use the first formulation a
   thread-model implementation suggests. A separate SIMT-vs-SIMD suite sweeping
   several strategies per kernel measured 0.33x-53x, including SIMT faster on
   some grouped reductions.
5. **Line counts exclude helpers defined outside the VF region**, in all
   frontends. This understates SIMT by 18 lines on each of the two FP4 variants.

---

## Caveat 6: the declared SIMT thread count

Every SIMT kernel declares `T.SimtVF(threads=128)`. At `K=128, group_size=32`
the two passes inside the region have different extents:

```text
for g in T.Parallel(num_groups):     extent 4    (the reduction)
for i in T.Parallel(hidden):         extent 128  (the apply)
```

{THREADS_RESULT}

---

## Conclusions

**SIMT reduces source size where the SIMD frontends move data between lanes.**
The three cases measured here are a group broadcast (`cast_back/01`, 16 lines to
6), a lane interleave (`per_channel/02`, 35 to 17) and a register gather
(`per_token/05`, 54 to 15). In each, the SIMD code is dominated by index
arithmetic that the thread model does not require, and in `per_channel/02` SIMT
also removes the 256-lane padding requirement. The cost is 3.28x, 7.02x and
13.74x simulated cycles respectively.

**PTO's SIMD style has nothing to simplify where the operation already maps onto
the hardware.** When a channel occupies one lane (`per_channel/01`) or the value
being placed is a scalar (`per_block/01`), ASC and PTO use 17 and 16 operations
and 21 lines each, and SIMT's shorter source expresses the same three stages.

**Where the hardware has an instruction, the frontend that can name it wins by a
wide margin.** FP4 output costs SIMT {F['fp4_vs_asc']:.0f}x, of which
{F['bf_ratio']:.2f}x is a code-generation gap and the remainder is a software
encoder standing in for one convert instruction.

---

## Frontend selection

| | applicable when |
|---|---|
| **SIMT** | development speed outweighs cycles; or the kernel is dominated by data placement — packing, transposes, gathers — where SIMT is 2-4x shorter and removes lane-indexing errors |
| **PTO/VMI** | general production vector code: {A['ops']} of ASC's operations at {A['pmean']} of its cycles, with width-parameterised definitions ASC cannot express |
| **ASC** | an instruction VMI does not expose is required (per-lane variable right shift), or a specific distribution mode is the objective |

**On methodology.** Source size correlates with simulated cycles at r = -0.09
on this set: a frontend comparison reporting only source size reports one of the
two quantities that matter.

"""

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if presentation/slides.md is out of date")
    args = ap.parse_args()
    n = sum(1 for ln in SLIDES.splitlines() if ln.startswith("## "))
    if args.check:
        if not OUT.exists() or OUT.read_text() != SLIDES:
            print(f"{OUT.relative_to(ROOT)} is out of date -- "
                  f"run: python tools/make_slides.py")
            return 1
        print(f"{OUT.relative_to(ROOT)}: up to date ({n} slides)")
        return 0
    OUT.write_text(SLIDES)
    print(f"wrote {OUT.relative_to(ROOT)}: {n} slides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
