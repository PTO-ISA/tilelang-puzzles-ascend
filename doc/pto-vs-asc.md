# PTO (VMI) vs ASC: what logical vector IR actually buys

Every PTO kernel file carries its own **PTO vs ASC** section with the specifics.
This document is the overview, and it is written to be falsifiable: the claims come
with measurements, and the variants where PTO wins nothing are listed too.

## The two surfaces

Both are vector IRs for the same chip, both live inside `with T.SimdVF():`, and in
tilelang they differ only by namespace and jit target:

```python
from tilelang.ascend.language import simd as S     # @tilelang.jit(target="ascend")
from tilelang.ascend.language import vmi  as V     # @tilelang.jit(target="pto")
```

ASC names *physical* operations: a distribution pattern from a fixed catalogue
(`BRC_B32`, `UNPK4_B8`, `PK4_B32`), a lane count baked into a dtype string
(`"float32x64"`), a predicate pattern by name (`"PAT_VL32"`).

VMI names the *intent* and lets the assembler choose the encoding: `size=` as an
argument, `group=` for segmented behaviour, `dist_mode="brc"`, `create_mask(32,
size=128)`.

## The evidence at production scale

[TileKernels-PTO](https://github.com/PTO-ISA/TileKernels-PTO) commit `5395526` ("Port four quant kernels to PTO/VMI") is the
whole argument in one diff. It touches exactly the four `*_asc.py` quant kernels,
rewriting only their `T.SimdVF` bodies — every `@T.prim_func` schedule and host
contract comes through byte-identical — and comes to:

```
312 insertions, 395 deletions      net -83 lines
```

That the schedules were untouched is as important as the line count: **the ASC/VMI
choice is a choice of vector instruction set, not of programming model.** Tiling,
buffering, DMA and core assignment are shared.

## What VMI actually removes

Four things, and all four are the same underlying idea — *the segment width or the
vector width becomes an argument instead of being baked into the instruction or the
type*.

### 1. `group=` replaces mask-and-select

A quant group is 32 channels; a float32 register is 64 lanes. ASC must reduce half
a register at a time and store the results one at a time:

```python
# ASC, 8 operations per 128 channels
S.vsts(amax_ub[g],     S.vcmax(a0, mask_low),  dist="ONEPT_B32")
S.vsts(amax_ub[g + 1], S.vcmax(a0, mask_high), dist="ONEPT_B32")
S.vsts(amax_ub[g + 2], S.vcmax(a1, mask_low),  dist="ONEPT_B32")
S.vsts(amax_ub[g + 3], S.vcmax(a1, mask_high), dist="ONEPT_B32")
```

```python
# PTO, 3 operations — "this vector is 4 independent segments; reduce each"
x = V.vcvt(V.vload(x_ub[col], size=128), "float32")
V.vstore(V.vcmax(V.vabs(x, mask), mask, group=4), amax_ub[group])
```

The same `group=` works on **loads** (segmented broadcast), **stores**, and
**broadcasts**, so one concept covers what ASC exposes as several unrelated
features. `per_token/01` measures 39 ASC operations against 19 for PTO.

### 2. Width is an argument, not part of the type

ASC reinterprets through dtype strings that fix the lane count — `"uint32x64"`,
`"float32x64"` — so the *same six-operation scale computation at a different width
is different code*. VMI derives the count from the total bit width, which means the
computation can be factored into a reusable helper:

```python
@T.macro
def compute_scale(amax, lanes):      # serves 4, 64, 128 and 256 lanes
    ...
```

Production's PTO `per_token` calls one such helper at four widths; the ASC version
cannot and spells the arithmetic out per path. That is a structural difference — it
changes what can be factored out — and it is where much of the 83 lines went.

### 3. Conversions are conversions

ASC expresses widening and narrowing as *distribution modes* you have to recognise
(`UNPK4_B8` = widen, `PK4_B32` = narrow), and expresses float↔float changes as bit
manipulation (`vdintlv` keep-high is a truncating fp32→bf16; `vintlv(zero, x)` is
bf16→fp32). VMI has `vcvt` with explicit `rounding=` and `saturate=`, and infers
packing from the destination buffer's dtype.

The widest single gap in this repo is `per_token/07`, production's bfloat16 fast
path. ASC needs group maxima of 32 bfloat16 values; 32 bf16 is 64 bytes, neither
one lane group nor one register, so there is no instruction for it. Production's
workaround is seven operations whose only purpose is to reach a grouping the
hardware *does* have — deinterleave 256 into evens and odds, pair the lanes with
`vmax` so each covers two originals, `vcgmax` over 16 lanes per group, then
zero-interleave the 8 results back to float32. VMI asks for the grouping it wants:

```python
amax = V.vcmax(abs_u, mask, group=8)      # 8 segments of 32. Done.
```

Four operations against sixteen.

### 4. Masks are counts, not catalogue entries

`S.pset(32, "PAT_VL32")` is a 64-lane hardware pattern; there is no 128-lane form
of it. `V.create_mask(32, size=128, group=2)` composes to any width.

Note a porting hazard: **`vsel`'s argument order differs.** ASC is
`S.vsel(if_true, if_false, mask)`, VMI is `V.vsel(mask, if_true, if_false)`. Getting
it wrong produces wrong numbers, not an error.

## Where VMI buys nothing, and where it is worse

This is the part that makes the rest credible.

`python tools/vf_lines.py` prints static vector-operation counts per variant.
Across all 22 paired variants PTO sits at roughly **three quarters** of ASC's
operation count — but it is concentrated, not uniform:

- **Large wins** — `per_token` throughout (`group=` replacing mask-and-select),
  `cast_back/05` and `per_token/07` (width, FP4 and bfloat16).
- **Draws** — `cast_back/02`, `cast_back/04`, `cast_back/06`, `per_token/05`, and
  essentially all of `per_block` and `per_channel`.
- **PTO slightly longer** — several `per_block` variants, because VMI requires
  explicit `size=` and mask operands and those variants have nothing to factor out.

The rule that explains the pattern: **VMI pays where ASC had to emulate something
the hardware does not directly offer.** It pays nothing where ASC was already
saying exactly what it meant — a plain contiguous load, a gather, a whole-register
reduce.

Genuine regressions:

- **No per-lane variable shift.** ASC can build a vector of shift amounts and apply
  `S.vshr(v, shifts)`. `V.vshrs` takes a scalar amount only, so production's PTO
  `cast_back` computes both byte positions and selects with `vcmp` + `vsel` where
  the ASC path did one shift.
- **No scalar-operand forms for some ops** (`vmul`, `vsub`), so constants must be
  broadcast into vectors first.
- **Mandatory `size=` and masks** make each call wider. Across the ladder PTO's
  *source line* count is about 90% of ASC's while its *operation* count is 74% —
  the gap between those two numbers is the ergonomic price.

### A measurement caveat worth stating

Both counts are **static** — operations written, not executed. Where PTO uses a
wider vector it also runs fewer loop iterations, and that does not appear in the
table. `per_block` is the clear case: PTO's static count is slightly higher, but
its reduction loop is 8 iterations of 128 lanes against ASC's 16 of 64. A static
count cannot see that, so read the per-variant docs rather than only the totals.

## The honest summary

VMI is **not** torch and not `T.Parallel`. You still open a vector scope, write
memory barriers by hand, live inside explicit Unified Buffer allocations, and spell
every width. What it removes is the *width and segment bookkeeping* — the part of
ASC code that exists only because the algorithm's natural granularity and the
register's geometry disagree.

"Torch-like vector operations inside an NPU schedule" is the fair description. For
a quantization kernel, where the granularity mismatch is the central difficulty,
that turns out to be most of the work.
