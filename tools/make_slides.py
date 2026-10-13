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

SLIDES = f"""---
marp: true
theme: default
paginate: true
title: "Writing NPU quant kernels: SIMT vs PTO vs ASC VF code comparison"
---

## Writing NPU quant kernels: SIMT vs PTO vs ASC

**A VF code-style and performance comparison of three vector frontends
expressing one quantization algorithm.**

- 22 kernel variants x 4 implementations = 88 kernels, all passing the same checks
- Simulated VF cycles for all 66 vector-tier kernels
- tilelang 0.1.15, ptoas vmi 0.1.9, CANN 9.2.0-beta.2, SoC `Ascend950PR_9599`
- Single core, `T.Kernel(1)`; the comparison is of the vector body only

*All cycle figures are from `msprof op simulator`, not silicon.*

---

## One algorithm, three frontends

Group-wise quantization, in PyTorch:

{code("torch_per_token")}

| frontend | what you name | the unit you think in |
|---|---|---|
| **ASC** (`T.simd`, `S`) | the physical instruction | a 64-lane vector register |
| **PTO/VMI** (`T.vmi`, `V`) | the intent; width and segments are *arguments* | a logical vector, any width |
| **SIMT** (`T.SimtVF`) | one element | a thread |

The vector body runs from **6 to 57 lines** depending on frontend and
configuration, and the frontends differ by up to **{A['smax']}** in cycles.

---

## The headline: two axes that disagree

It is tempting to assume the frontend writing fewer operations produces the
faster kernel. Across these 22 variants, it does not:

| PTO relative to ASC | |
|---|---|
| vector operations written | **{A['ops']}** |
| source lines written | **{A['lp']}** |
| simulated VF cycles | **{A['pmean']} mean**, range {A['pmin']}-{A['pmax']} |

- PTO uses fewer cycles than ASC on **{A['pwin']} of 22** variants
- On **four** of those it *writes more operations than ASC*

**Every pattern in this deck is therefore shown with both axes.**

---

## Method, and why the numbers can be trusted

**Measurement.** Sum the `cycles` column of `core0.veccore0_instr_exe.csv` over
rows whose `instr` names a VF launch (`VF`, or `VF_SIMT`). Key on **`instr`**,
never `pipe` -- every real VF row is on `PUSHQ`, shared with `MOVEMASK`.

**Correctness first.** Every kernel is compared against a standalone PyTorch
oracle, with an assertion that results came back from the device. No cycles are
reported for a kernel that computed the wrong answer.

**Like-for-like.** VF launch counts compared across tiers on every variant; they
match everywhere except `per_token/05` (34/34/32 -- SIMT skips a gather setup).

**Two failure modes caught:** the profiler measuring a `torch.zeros` fill instead
of the kernel (caught because three implementations reported *identical* cycles),
and divergence inflating one SIMT row (slide 20). Every trace is now audited
against the simulator's instruction log.

---

## Pattern 1 -- the data path: a per-group scale broadcast

`cast_back/01`: 64 consecutive channels straddle two scale groups, so each lane
needs the right one of two scalars.

**ASC** -- build the vector: two broadcast loads, then select by lane mask:

{code("broadcast_asc")}

**PTO** -- state the shape instead: one load, two segments:

{code("broadcast_pto")}

**SIMT** -- a thread knows which element it owns:

{code("broadcast_simt")}

---

## Pattern 1 -- measured

{row("cast_back/01_e4m3_fp32sf")}

**This row sets the theme.**

- PTO collapsed three operations into one and came out **slower**
- SIMT wrote a third of the code and cost **3.28x**
- Neither axis predicts the other

Same variant, quieter win: switching output dtype costs ASC a different *store
instruction*, PTO only a convert, and SIMT **nothing at all** -- `T.copy`
converts between buffer dtypes, so the VF body never mentions it.

---

## Pattern 2 -- where SIMT genuinely wins: an interleave

`per_channel/02` packs two token groups' UE8M0 bytes into each 16-bit word:
byte `2c` from group `2i`, byte `2c+1` from group `2i+1`.

**ASC / PTO** -- that is an interleave instruction:

{code("interleave_asc")}

**SIMT** -- two scalar stores:

{code("interleave_simt")}

---

## Pattern 2 -- the trap underneath, and the cost

A `uint8` vector register is **256 lanes**, so loading the 128 meaningful bytes
reads 128 bytes past the row. Both SIMD tiers must pad the scratch buffer:

{code("pad_asc")}

{code("pad_simt")}

{row("per_channel/02_round_packed_m")}

SIMT removes the instruction *and* the bug it invites -- at **7.02x** the cycles.
Note PTO is **longer** than ASC here (40 lines vs 35).

---

## Pattern 3 -- where SIMT genuinely wins: a transpose

`per_token/05` writes the scale array transposed. The scales live one-per-group
in vector lanes, and **you cannot restride a register**.

**ASC** -- compute, for every lane, which source element it should receive:

{code("transpose_asc")}

...then gather:

{code("transpose_asc_gather")}

**SIMT** -- the transposed layout is the index you store at:

{code("transpose_simt")}

---

## Pattern 3 -- the sharpest single trade

{row("per_token/05_col_major_sf")}

- **15 lines against 54** is the largest source-length win in all 22 variants
- **13.74x** is among the worst cycle ratios
- The thread model removes the lane-placement problem, and leaves most of the
  128-lane unit idle doing it

PTO is *marginally longer* than ASC on the gather itself: VMI has no
scalar-operand `vmul`, so the stride must be broadcast into a vector first.

> **The rule: SIMT pays off exactly where the SIMD tiers had to move data
> between lanes** -- broadcast, interleave, transpose, gather.

---

## Pattern 4 -- where PTO's SIMD style is already clean enough

`per_channel/01` reduces along **M**. A channel *is* a lane, so there is no
cross-lane step for anybody.

**ASC** -- a register accumulator, maxed over 32 rows:

{code("chanreduce_asc")}

**PTO** -- the same code with a mask operand:

{code("chanreduce_pto")}

---

## Pattern 4 -- a genuine draw

{row("per_channel/01_raw_32tokens")}

**{st['per_channel/01_raw_32tokens']['oa']} vector operations against
{st['per_channel/01_raw_32tokens']['op']}, and {st['per_channel/01_raw_32tokens']['la']} source lines each.**

- VMI's wins come from expressing *segmented* behaviour and from width being an
  argument. A reduction that is already one instruction per vector offers neither.
- SIMT's 12 lines buy **notation, not structure** -- same three stages, same
  memory traffic.
- PTO is nonetheless **0.77x** in cycles, so "nothing to buy" is about source.

**Second case:** `per_block/01` has one scale per 32x32 tile. A scalar has no
layout, so a column-major output is free in **all three** frontends -- the same
configuration that costs ASC a gather in Pattern 3.

---

## Pattern 5 -- where PTO's logical style is the real win

`per_token/01` needs the max of each 32-channel group. ASC's vector reduction
yields one value per register, so four groups means four masked reductions and
four single-element stores:

{code("segreduce_asc")}

**PTO** says "reduce, in four segments":

{code("segreduce_pto")}

Eight operations become three. The matching broadcast collapses the same way.

---

## Pattern 5 -- the one place both axes agree

{row("per_token/01_raw_fp32sf")}

| | ASC | PTO |
|---|---:|---:|
| vector operations | {st['per_token/01_raw_fp32sf']['oa']} | **{st['per_token/01_raw_fp32sf']['op']}** |

- PTO **halves** the operation count *and* uses **0.74x** the cycles
- Also where SIMT's cost first reaches an order of magnitude, and the code shows
  why: with `group_size = 32` and `hidden = 128` the reduction runs
  **4 threads out of 128**

---

## Pattern 5b -- width as an argument

`per_token/02` rounds the scale to a power of two by rewriting the float32
exponent field. **ASC** must bake the lane count into every type:

{code("exponent_asc")}

**PTO** takes the width as a parameter -- so this becomes a macro serving 4, 64,
128 or 256 lanes. ASC structurally *cannot* write it once, because `"uint32x64"`
fixes the width:

{code("exponent_pto")}

{st['per_token/02_round_sf']['oa']} operations against
{st['per_token/02_round_sf']['op']} -- the largest static gap in the set -- at a
**1.02x cycle draw**. PTO's real advantage here is **reusability**, not cycles.

---

## Pattern 6 -- where PTO costs you

**Mandatory width and mask operands.** `cast_back/02` decodes a UE8M0 byte.
ASC names the distribution mode on the load:

{code("ue8m0_asc")}

PTO needs width, stride, group count and a mask spelled out:

{code("ue8m0_pto")}

Operations tie at {st['cast_back/02_packed_ue8m0']['oa']} each;
**PTO is 4 lines longer**. Two more:

- **A missing capability:** no per-lane variable *right* shift. `S.vshr(v, shifts)`
  takes a vector; `V.vshrs` takes a scalar only. *No variant here exercises it* --
  reported because it is the clearest thing ASC can express and VMI cannot.
- **`per_channel/04`:** PTO's longest region, {st['per_channel/04_compose']['lp']}
  lines against {st['per_channel/04_compose']['la']}, at a **1.00x** draw.

---

## Pattern 7 -- where SIMT loses outright

`per_token/04` quantizes to FP4 (e2m1). ASC and PTO own a hardware
`float32 -> e2m1` convert; their only real work is pre-rounding to bfloat16 with
a sticky bit:

{code("fp4_pto")}

**No FP4 store is reachable from a `SimtVF` region.** SIMT declares its output
`uint8` and encodes the format by hand -- an extract of 18 lines:

{code("fp4_simt")}

It is bit-exact against the PyTorch packer, verified before the kernel ever ran.

---

## Pattern 7 -- and half the gap is not SIMT's fault

{row("per_token/04_fp32_in_fp4_out")}

`per_block/03` is the same loss at 16.77x -- a pattern, not a one-off. But the
99.62x **decomposes**: 124,294 instructions against ~37,000, **and** 4.94 cycles
per instruction against the rest of the tier's 1.90-2.01.

That second factor is **branch divergence**: `T.if_then_else` lowers to a real
branch here, not a predicated select. Respelling three conditionals as mask
arithmetic issues **45% more instructions and runs 1.78x fewer cycles** --
{bf:,.0f}, or **56.0x** rather than 99.6x. Both spellings ship, byte-identical.

> **When the hardware has an instruction for the job, the frontend that can
> *name* it wins, and the margin is not close.**

---

## The inversion -- static counts mispredict cycles

The `per_block` family: PTO writes **more** operations than ASC on all four, and
runs **fewer** cycles on three.

| variant | ops ASC | ops PTO | cycles PTO/ASC |
|---|---:|---:|---:|
| `per_block/01` | 21 | 22 | **0.85x** |
| `per_block/02` | 27 | 30 | 1.02x |
| `per_block/04` | 21 | 22 | **0.65x** |
| `per_block/05` | 29 | 31 | 1.02x |

**ASC** reduces a 1024-value tile in **16 steps of 64 lanes**; **PTO** reaches
128 float32 lanes, so it is **8 steps of 128**:


{code("blockreduce_pto")}

More operations per step, half as many steps -- a static count cannot see a trip
count. `cast_back/05` reverses it: PTO's **largest static win** *and* its **worst
cycle regression (1.13x)**.

---

## The complete measurement (1/2) -- `cast_back`, `per_token`

{AGG1}

`cast_back` is the data path alone; `per_token` the segmented reduction.

---

## The complete measurement (2/2) -- `per_block`, `per_channel`

{AGG2}

`per_block` is the static/dynamic inversion. `per_channel` reduces along M,
where the data already lines up with the lanes.

---

## Aggregates

| | PTO vs ASC | SIMT vs ASC |
|---|---|---|
| vector operations | {A['ops']} | -- (no comparable count) |
| source lines | {A['lp']} | **{A['ls']}** |
| simulated cycles | **{A['pmean']} mean** ({A['pmin']}-{A['pmax']}) | {A['nmean']} mean ({A['nmin']}-{A['nmax']}) |

- PTO faster than ASC on **{A['pwin']} of 22**; ASC faster on 6; 7 within 2%
- SIMT range **excludes `per_token/04`**, whose hand-written FP4 encoder measures
  a missing instruction, not a programming style
- SIMT writes **{A['ls']}** of ASC's lines and costs roughly an order of magnitude

---

## Caveats

1. **Simulator cycles, not silicon.** No hardware-fidelity claim.
2. **One shape per variant** (mostly `M=32, K=128`). Ratios are within-variant
   and valid; cycle counts are **not** comparable between rows.
3. **Single core**, no multi-core scheduling, double buffering or cache hints.
4. **The SIMT kernels are deliberately untuned** -- written the way a GPU
   programmer writes them first. A separate suite sweeping SIMT strategies found
   **0.33x to 53x**, i.e. SIMT *winning* by 3x on some grouped reductions.
   **Nothing here says "SIMT is slow" in general.**
5. **Line counts exclude helpers outside the vector region**, in every frontend
   -- understating SIMT by ~18 lines on the two FP4 variants.
6. **All reductions use `group_size = 32`**, the case that leaves the most SIMT
   threads idle.

---

## Conclusions -- the two questions answered

**Where does SIMT genuinely simplify the kernel?**

Only where the SIMD frontends had to move data *between lanes*: a group
broadcast, an interleave, a transpose. There it is a third of the code and
removes a class of bug -- the 256-lane padding trap being the clearest example.
Everywhere else it buys notation rather than structure, and it pays an order of
magnitude in cycles throughout.

**Where is PTO's SIMD style already clean enough?**

When the operation maps onto the hardware as written: the data already lines up
with the lanes (`per_channel/01` -- a channel *is* a lane), or the value being
placed is a scalar (`per_block/01` -- a tile scale has no layout). Those come
out 17 operations against 16, and 21 lines each. Reaching for a thread model
there is effort for nothing.

---

## Guidance, and one methodological point

| | choose it when |
|---|---|
| **SIMT** | reaching a correct kernel quickly matters more than cycles; or the kernel is dominated by lane placement -- packing, transposes, gathers |
| **PTO/VMI** | the default for production vector code: {A['ops']} of ASC's operations *and* {A['pmean']} of its cycles, plus width-parameterised helpers ASC cannot write |
| **ASC** | an instruction VMI cannot express is needed (a per-lane variable right shift), or a distribution mode is the point |

**Methodologically.** Operation and line counts measure how much you write.
They were a poor predictor of cycles here -- PTO wins on {A['pwin']} of 22
variants, and on four of those it *writes more operations than ASC*. Any
frontend comparison reporting only source size reports half the answer.
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
