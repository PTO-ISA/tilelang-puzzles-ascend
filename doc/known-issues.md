# Known issues and toolchain limits

Measured on **tilelang 0.1.15** + **ptoas vmi 0.1.9** + **CANN 9.2.0-beta.2**,
2026-10-07, in the container built by `docker/Dockerfile`.

Everything here is reproducible by a script in `harness/probe/`. Re-run those after
any toolchain upgrade; if one starts passing, the corresponding workaround can be
removed.

```bash
python harness/probe/vf_lane_limits.py          # compile-only, ~10 s
python harness/probe/fp8_out_idx.py             # needs the simulator
python harness/probe/simt_limits.py             # compile-only, ~1 min
python harness/probe/simt_limits.py --run       # + numerical check, under the simulator
python harness/probe/simt_divergence.py         # a performance cliff, ~16 min
```

## Hard limits (worked around in this repo)

### 8-lane bfloat16 → float32 convert does not lower

```
VMI-LAYOUT-CONTRACT: pto.vmi.extf has no registered layout support:
  source/result layouts do not match a legal cast table row; VMI types:
  operand#0=!pto.vmi.vreg<8xbf16, #pto.vmi.layout<contiguous>>
  result#0=!pto.vmi.vreg<8xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

Reproduce: `harness/probe/vf_lane_limits.py`, case `per_block_8lane`.

This matters because a 32×32 tile is 32 values wide and 32 is **not** a legal
vector length (the allowlist is `{1, 2, 4, 8, 64, 128, 256}` — see
`vf-lane-widths-and-limits.md`), so 8 lanes is the natural next guess. It does not
work.

**Workaround, used by `per_block` on both backends and by production:** flatten the
tile and reduce at 64 or 128 lanes. The general rule is to pick the lane width from
the hardware and reshape the problem to fit, rather than following the data's own
geometry.

### tilelang's `out_idx` cannot allocate a float8 output

```
MemoryError: Unsupported code 10
```

Reproduce: `harness/probe/fp8_out_idx.py`.

torch allocates `float8_e4m3fn` on the device fine, and passing an explicitly
allocated float8 tensor to the kernel works. Only the `out_idx` auto-allocation
path fails.

**Workaround:** every quantize kernel (`per_token`, `per_block`, `per_channel`)
allocates its FP8/FP4 output in `launch()` and passes it in. That is also how the
production kernels are called, so it costs nothing pedagogically. `cast_back` is
unaffected — its outputs are bfloat16 or float32.

### `S.vmins` has no uint16 overload

```
error: no matching function for call to 'asc_min_scalar'
```

Hit by the FP4 round-to-odd sticky bit, which wants `min(low, 1)` on a uint16
vector. Production uses the scalar-operand spelling; it does not compile here.

**Workaround:** `S.vmin` against a splatted `1`. One extra `vdup`, identical
results (`per_token/04` matches the bit-exact torch packer on 4096/4096 codes).

### No float32 → e2m1 conversion

```
ASC: ValueError: Unsupported vcvt conversion float32->float4_e2m1fn
PTO: TypeError: T.vmi.vcvt(...) supports packed FP4 only for bfloat16 to float4_e2m1fn
```

Not a bug — there is no such instruction. FP4 output must go
float32 → bfloat16 → e2m1, and the intermediate must be **rounded to odd** or
double rounding biases the result. See `per_token/04`.

### No FP4 value exists inside a `T.SimtVF` region

```
InternalError: Check failed: index_lanes * buffer_lanes == value_dtype_lanes
(2 vs. 1) : Cannot store value with 1, expected value with 2
(1 index lanes * 2 buffer element lanes)
```

The entry above is about the *conversion*; this is about there being nothing to
convert into. `harness/probe/simt_fp4_convert.py` tried nine spellings — a
scalar `T.Cast` to `float4_e2m1fn`, a `T.vectorized(2)` pair, an fp4 fragment at
2, 4, 8 and 16 codes per thread, a `T.copy` of a float32 fragment into fp4
storage, an explicit `T.Cast("float4_e2m1fnx2", ...)`, and a reinterpret of a
code as a byte. All nine fail before codegen, with three errors: the one above,
`Cannot match type float4_e2m1fnx2 vs float4_e2m1fnx4`, and
`reinterpret expects source and target to have the same number of bits`.

The cause is not the frontend. An fp4 element is **four bits**, so fp4 storage
is retyped to `float4_e2m1fnx2` — one byte, two codes — before Ascend codegen
(`src/ascend/transform/rewrite_fp4_to_fp4x2.cc`), and every store into it needs
`index_lanes * 2` lanes. SIMT code is per-thread *scalar* code with one lane to
offer. Underneath, the toolchain agrees: in `include/pto/npu/a5/` the only FP4
types are `vector_f4e2m1x2` / `float4_e2m1x2_t`, and the conversion is
`vcvt(vector_f4e2m1x2, vector_bf16, preg, ROUND_R, PART_P0)` — a vector
instruction over a vector-only type. **There is no scalar FP4 type**, which is a
property of the hardware's data types rather than a gap to be filled.

An upstream bug report for this, self-contained and ready to file, is
[`issues/ascend-simt-fp4-cast.md`](issues/ascend-simt-fp4-cast.md): it carries a
single compile-only reproduction script, the three internal errors it raises,
the framework locations they come from, and the contrast with the CUDA backend,
which does have a scalar FP4 type.

**Workaround 1, and what the ladder ships:** declare the output `uint8` and
encode the nibbles by hand, one output **byte** per thread so the pack needs no
exchange between threads. Bit-exact against the torch packer, 18 lines against
PTO's 4-operation round-to-odd, and the honest price of staying inside the
thread model — see `per_token/04` and `per_block/03`, and `doc/tiers/simt.md`.

**Workaround 2, and the cheap one:** leave the region for that one instruction.
A `T.SimtVF` region and a `T.SimdVF` region can live in the same kernel, so the
thread model can do the reduction, the scales and the round-to-odd, and three
lines of `T.SimdVF` can do the convert:

```python
with T.SimdVF():
    for pair in T.serial(hidden // PAIR):
        S.vsts(q_ub[pair * PAIR],
               S.vcvt(S.vld(bf_ub[pair * PAIR]), T.float4_e2m1fn),
               dist="PK4_B32")
```

Measured on `per_token/04`, both arms bit-exact against the torch packer and
byte-identical to each other: **543,827 cycles against 7,778** (6,050 `VF_SIMT`
plus 1,728 `VF`), a **69.9x** difference, and 1.26x ASC rather than 88x. It
ships in that variant behind `hw_convert=`, and the harness verifies it; it is
deliberately **not** the default, because the SIMT column measures what the
thread model costs on its own. `harness/probe/simt_fp4_convert.py` has both.

### bfloat16 does not compute inside a `T.SimtVF` region

```
with T.abs:    InternalError: Unresolved call ir.Op(... name="tirx.fabs" ...)
without:       RuntimeError: Ascend device compilation failed.
```

Both spellings of a bfloat16 accumulator fail, and `T.abs` on **float32** in the
same position compiles and is correct — so this is bfloat16, not `T.abs`.

**Workaround:** let `T.copy` widen bfloat16 to float32 on the way into the
fragment, compute in float32, and convert on the way out. For the quantization
kernels this costs no accuracy at all: the scale is a power of two, so scaling a
bfloat16 value by it is exact in either precision, and the scale depends only on
the *exponent* of the maximum, which bfloat16 represents exactly. The harness
asserts that second point directly (`per_token/07`, `per_channel/04`).

It does mean those two variants lose their *lesson* rather than their
correctness: both exist to show bfloat16 doubling the vector lane count, and the
thread model has no lane count to double.

### `cannsim` hangs on `vshr` / `vshl`

Every variant from `per_token/02` onwards uses the ceil-log2 exponent trick, which
needs vector shifts. Under `cannsim` / `npusim` those hang indefinitely.

**This repo therefore defaults to `msprof op simulator`** (`Ascend950PR_9599`).
`TLP_SIMULATOR=cannsim` still selects the old runner, but it cannot run most of
the ladder. This is not a regression to fix in the kernels — it is why the default
is what it is.

## Performance cliffs (correct, but much slower than they look)

### `T.if_then_else` lowers to a divergent branch, not a select

This one costs **2.09x** on `simt/per_token/04`, and it is the reason that
variant measures 91.3x ASC's simulated VF cycles while the rest of the SIMT
tier sits between 1.2x and 10.2x.

Measured on that variant, which ships both spellings and whose two arms the
harness asserts are byte-identical. Every column comes from the same place --
the kernel's own trace, counted by `tools/vf_trace_audit.py` over every
vector-pipe instruction line in `core0.veccore0.instr_log.dump`:

| spelling | cycles | instructions | cycles/instr | `SIMT_*_DVG` / `SIMT_BRANCH` |
|---|---:|---:|---:|---:|
| `T.if_then_else(c, a, b)` | 562,280 | 243,130 | 2.31 | 1,408 / 44,835 |
| `(a & m) \| (b & ~m)`, `m = -T.Cast("int32", c)` | **269,370** | **152,260** | **1.77** | **0 / 0** |

The branch-free form issues **37% fewer instructions and runs 2.09x fewer
cycles**: the branchy spelling's extra instructions *are* the branching --
44,835 `SIMT_BRANCH` and 22,788 integer predicate sets that the mask form does
not emit at all.

`harness/probe/simt_divergence.py` is the controlled experiment: it rebuilds the
encoder standalone and reproduces the mechanism (615,170 against 351,956 cycles,
5,864 divergence markers against 0). Its instruction counts do not match the
whole kernel's, and cannot: a divergent branch issues every path it takes, so the
conditional arm's count depends on its input, and the probe's input is not the
harness's. The mask arm has no branches and is the stable side of the
comparison.

The trace names the mechanism directly: `SIMT_START_DVG` / `SIMT_END_DVG`
markers appear only where `T.if_then_else` is used on a data-dependent
per-element condition, and `simt/per_token/04` is the only variant in the ladder
with a large number of them.

**This is a codegen gap, not a property of the thread model.** On a GPU,
`cond ? a : b` over cheap arithmetic is predicated into a select by the
compiler, so the natural conditional costs nothing; here it is a real branch.

**Workaround, if you need the cycles:** build a mask with
`m = -T.Cast("int32", cond)` — 0 or all-ones — and select with
`(a & m) | (b & ~m)`. Both arms must be safe to evaluate unconditionally, which
in the FP4 encoder they are, because the shift amount is already clamped.

**Both spellings ship.** `simt/per_token/04` takes a `branch_free` flag:
`launch(x)` is the conditional form and `launch(x, branch_free=True)` the mask
form. The harness checks both and asserts they are byte-identical, so neither is
an unverified aside, and `doc/quant_vf_cycle_comparison.md` publishes both cycle
counts.

The default stays the conditional on purpose: this ladder wants the spelling a
GPU programmer reaches for first, and the measurement is more useful when it
reports what that spelling actually costs. The published SIMT range quotes that
variant separately for the same reason.

### A single-thread `T.Parallel(1)` diverges too, by definition

`per_block/01`, `/02`, `/04` and `/05` each show 48 markers (32 `SIMT_*_DVG`
plus 16 `SIMT_BRANCH`): the one-thread pass that clamps the tile maximum and
forms the two scales, where 127 threads sit idle. The absolute cost is now
small -- those kernels run at 1.54-1.76 cycles per instruction, in line with
the rest of the tier.

It was **not** small when the whole reduction lived in that pass. These variants
originally folded 32 per-row partials there, which measured 68 markers, 7.14
cycles per instruction -- the worst in the tier -- and 19,276 cycles against
2,664 now. The lesson is the one `simt_reduce_max.py` makes on `per_token/01`:
put the work in a `T.Parallel` as wide as the data and let `T.reduce_absmax` do
the fold. A single-thread pass is fine for three scalar operations and ruinous
for a loop.

### `T.reduce_absmax(dim=0)` and a `uint8` store do not mix

`per_channel/02` and `/04` write their scale as a UE8M0 **byte**. Spelling their
column-wise fold as `T.reduce_absmax(x_l, amax_l, dim=0)` -- which is correct
and fractionally faster on `per_channel/01`, where the scale is float32 -- fails
to lower:

```
SimtVF cannot convert vector type uint8x2 to an Ascend type
```

Reduced reproductions do not fail: a `(32, 128)` fold followed by a `uint8`
store, with or without a 2-D destination and with or without the byte-pack pass,
compiles. So this is recorded as **observed in context, not diagnosed**, and it
is one of the reasons the whole `per_channel` family keeps the serial scan --
see [`per_channel/01`](quant/per_channel/01_raw_32tokens.md) for the other two,
which are that there is no thread occupancy to recover there and that the CUDA
reference scans the same axis the same way.

## Frontend rules that produce confusing errors

### SIMD values are immutable

An accumulator carried across loop iterations cannot be a plain value:

```
Immutable variable `acc` is used outside its defining region!
```

It must live in a register array — `S.alloc_local((1,), T.float32)` or
`V.alloc_local((1,), V.vreg(128, T.float32))`. The message does not say that.
Used in `per_block/01` and `per_channel/01`.

### `S.pset` must be bound to a name

```
tvm.error.InternalError: Unresolved call ir.Op(... name="tl.simd.pset" ...)
```

Nesting it inside another call fails:

```python
mask_high = S.pnot(mask_low, S.pset(32, "PAT_ALL"))    # breaks
mask_all  = S.pset(32, "PAT_ALL")                      # works
mask_high = S.pnot(mask_low, mask_all)
```

### `T.unroll`'s loop variable is symbolic

A Python-level condition on it is a `PrimExpr`, not a `bool`, so the branch
silently always takes its first arm:

```python
for half in T.unroll(2):
    values = x_lo if half == 0 else x_hi      # always picks x_lo. No error.
```

Unroll in Python when the body needs a trace-time decision:

```python
for half, values in enumerate((x_lo, x_hi)):  # correct
```

This cost a wrong-answer debugging pass in `cast_back/06` — the output was close
to correct, which is the hard kind of wrong.

### A uint8 vector load is 256 lanes

`S.vld(..., dist="NORM_B8")` reads a whole 256-byte register. Loading a 128-byte
scale row therefore reads 128 bytes past its end. `per_channel/02` pads its scale
rows for exactly this reason; without the padding, 75 of 256 output bytes were
wrong — again, mostly right.

### One fragment, one index pattern per `T.Parallel` body

```
InternalError: Check failed: (StructuralEqual()(it->second.indices, indices))
is false: v: (p * 2 + 1,) and (p * 2,)
```

Inside a single `T.Parallel` body a **fragment** may be touched with exactly one
structurally identical index expression (`RecordBufferAccess` in
`src/op/parallel.cc`). Reading `v[2 * p]` and `v[2 * p + 1]` in the same body is
rejected. **Shared and UB buffers are exempt**, and the probe checks both halves
of that.

This is the single rule that decides what SIMT code in this repo looks like:
irregular, packed or data-dependent index math lives on `alloc_shared`, and
fragments carry only the clean subscripts of the `Parallel` variables. Where a
SIMT kernel here stages intermediates in a shared scratch buffer —
`per_token/06`, `per_channel/03`, `per_block`'s partials — this is why.

### `T.copy` into a 2-D buffer copies a 2-D region

Not a bug, and the most expensive confusion in building the SIMT tier, because it
is silent. With `ub` of shape `(4, 32)`:

```python
T.copy(X[token, 0], ub)      # NOT X[token, 0:128] reshaped
                             # it is the region X[token:token+4, 0:32]
```

On numbered input the probe prints `got[0][:4] = [128.0, 129.0, 130.0, 131.0]`,
which is `X[1, 0:4]` — so the first group is correct and the rest hold *other
tokens'* data. In a per-token reduction that reads exactly like a broken
reduction, and it cost two wrong diagnoses before the numbers were dumped.

**Workaround:** keep UB staging buffers **flat** and put the group structure in
the index arithmetic, which shared buffers allow freely. A 2-D UB tile is right
when the tile really is 2-D — `per_block` (32 tokens by 32 channels) and
`per_channel/03`'s scale block both want exactly the region semantics.

### Widths must match exactly in VMI

`V.vsub(a, b)` with `a` at 1 lane and `b` at 128 raises
`T.vmi.vsub(...) requires identical VMI vector types`. Constants have to be built
at the width they will be used at, which is why `per_block/05` creates `size=1`
constants for its single-scale arithmetic.

## What this list does *not* contain

No variant in this repo falls back to computing its result on the host. Every one
of the 88 kernels runs on the device and is checked against the torch tier;
`harness/status.py` asserts that outputs come back from `npu`, so a host fallback
cannot hide behind a `PASS`.

In particular, two bodies that are easy to assume are unsupported do work on this
pin, and `harness/probe/vf_lane_limits.py` re-checks both on every run:

- a **fused 128-lane** per_token body — one vector carried through convert,
  segmented reduce, divide and segmented broadcast, with no bounce through
  Unified Buffer between the divide and the broadcast;
- the same body with a **float32 input**.

Which bodies lower is a property of the toolchain version, not of the hardware, so
treat that probe's output as the answer rather than this document.
