# Writing NPU quant kernels: SIMT vs PTO vs ASC

**A code-style and performance comparison of three vector frontends expressing one
quantization algorithm.**

Measured on tilelang 0.1.15 + ptoas vmi 0.1.9 + CANN 9.2.0-beta.2, SoC
`Ascend950PR_9599`, under `msprof op simulator`. All numbers in this report are
simulated VF cycles from that model, not silicon measurements.

---

## 1. The question

Group-wise quantization is a small algorithm. For each group of 32 channels, take
the maximum absolute value, turn it into a scale, and divide through:

*source:* `puzzles/torch/quant/answer/per_token/01_raw_fp32sf.py:23-26`

<!-- snippet: torch_per_token -->
```python
grouped = x.float().view(m, k // group_size, group_size)
amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
sf = amax / E4M3_MAX
q = (grouped * (E4M3_MAX / amax).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
```

Two or three lines of PyTorch. On an Ascend vector unit the same algorithm can be
written three ways. Depending on the frontend and the configuration, the vector
body runs from **6 to 57 lines**, and the three frontends differ by as
much as **two orders of magnitude** in simulated cycles. This report asks which
of those differences are real, and which are an artefact of what we chose to
count.

The three frontends, all targeting the same chip and the same single vector core:

| frontend | abstraction level | smallest unit of data |
|---|---|---|
| **ASC** (`T.simd`, imported `S`) | named hardware instruction and distribution mode | one 64-lane vector register |
| **PTO/VMI** (`T.vmi`, imported `V`) | operation with width and segment count as arguments | one logical vector, any width |
| **SIMT** (`T.SimtVF` + `T.Parallel`) | per-element expression | one element, held by one thread |

The comparison rests on 22 kernel variants across four quantization kernels
(`cast_back`, `per_token`, `per_block`, `per_channel`), each implemented in all
three styles plus a PyTorch reference — 88 implementations, all passing the same
correctness checks against the same oracle. This report shows eight of the 22,
chosen because each isolates one mechanism. The rest corroborate.

**Out of scope**, deliberately: multi-core scheduling, double-buffered Unified
Buffer, and cache hints. Every kernel here is single-core `T.Kernel(1)`, so the
comparison is of the `T.SimdVF` / `T.SimtVF` region and nothing else.

### The result that shapes the rest of this report

It is tempting to assume that the frontend which writes fewer operations
produces the faster kernel. Across these 22 variants that assumption is wrong
often enough to be the most important finding:

| PTO relative to ASC | |
|---|---|
| vector operations written | **76%** |
| source lines written | **94%** |
| simulated VF cycles | **92% mean**, range 0.64x–1.13x |

PTO uses fewer cycles than ASC on **9 of 22** variants — and on four of those it
*writes more operations than ASC*. Operations written and cycles executed
correlate at Pearson r = -0.09 across the 22 variants, so this report reports
both quantities for every pattern. Each pattern below is
presented with both.

---

## 2. Baseline: the pure data path

`cast_back` is the only kernel in the set with no reduction — the scale factors
are an input. It is therefore the cleanest place to see the three styles side by
side, with nothing but a load, a multiply, a convert and a store.

The algorithm:

*source:* `puzzles/torch/quant/answer/cast_back/01_e4m3_fp32sf.py:31-33`

<!-- snippet: torch_dequant -->
```python
grouped = q.float().view(m, k // group_size, group_size)
out = grouped * sf.unsqueeze(-1)
target = torch.bfloat16 if out_dtype == "bfloat16" else torch.float32
```

The one interesting part is the **broadcast**: 64 consecutive channels straddle
two scale groups, so each lane needs the right one of two scalars.

ASC must build that vector explicitly — load each scalar broadcast across all
lanes, then select between them with a lane mask:

*source:* `puzzles/asc/quant/answer/cast_back/01_e4m3_fp32sf.py:66-70`

<!-- snippet: broadcast_asc -->
```python
lo = S.vld(sf_ub[group], dist="BRC_B32")
hi = S.vld(sf_ub[group + 1], dist="BRC_B32")
scale = S.vsel(lo, hi, mask_low)

scaled = S.vmul(values, scale)
```

PTO states the shape instead of constructing it. A slot load plus grouped
broadcast describes two segments; PTOAS fuses the pair:

*source:* `puzzles/pto/quant/answer/cast_back/01_e4m3_fp32sf.py:60-64`

<!-- snippet: broadcast_pto -->
```python
slots = V.vload(sf_ub[group], size=groups_per_strip,
                stride=1, group=groups_per_strip)
scale = V.vbrc(slots, size=LANES, group=groups_per_strip)

scaled = V.vmul(values, scale, mask)
```

SIMT does not have the problem. A thread knows which element it owns, so the
scale is the value at an index:

*source:* `puzzles/simt/quant/answer/cast_back/01_e4m3_fp32sf.py:58-59`

<!-- snippet: broadcast_simt -->
```python
for i in T.Parallel(hidden):
    o_l[i] = q_l[i] * sf_ub[i // group_size]
```

| | ASC | PTO | SIMT |
|---|---:|---:|---:|
| VF-region lines | 16 | 12 | **6** |
| simulated cycles | **2,105** | 2,201 | 6,908 |
| vs ASC | 1.00x | 1.05x | 3.28x |

**This row sets the theme.** PTO collapsed three operations into one and came out
**1.05x slower**. SIMT replaced lane plumbing with an index expression, wrote a
third of the code, and cost **3.28x**. Neither axis predicts the other.

### A second, quieter difference

The same variant switches output dtype between `bfloat16` and `float32`. In ASC
the dtype selects the *store instruction*, and the intuition is inverted — the
wider dtype takes the simpler instruction:

*source:* `puzzles/asc/quant/answer/cast_back/01_e4m3_fp32sf.py:77-82`

<!-- snippet: outdtype_asc -->
```python
if out_dtype == "bfloat16":
    S.vsts(out_ub[col], S.vcvt(scaled, T.bfloat16),
           dist="PK_B32")      # 64 lanes -> 64 x 2 bytes
else:
    S.vsts(out_ub[col], scaled,
           dist="NORM_B32")    # 64 lanes -> 64 x 4 bytes
```

In PTO the destination buffer's type decides the packing, so only the convert
appears or disappears:

*source:* `puzzles/pto/quant/answer/cast_back/01_e4m3_fp32sf.py:71-74`

<!-- snippet: outdtype_pto -->
```python
if out_dtype == "bfloat16":
    V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
else:
    V.vstore(scaled, out_ub[col])
```

In SIMT this configuration costs **nothing at all**: `T.copy` converts between
buffers of different dtypes, so the VF body never mentions the output type. This
is a better example of PTO's advantage than the broadcast, because here the
saving is structural rather than three operations becoming one.
---

## 3. Irregular data placement: where SIMT reduces source size

If SIMT has a thesis, it is this section. Both cases below are ones where the
SIMD tiers must move data **between lanes**, and a vector register's lane
assignment is fixed.

### 3.1 An interleave: packing scales along M

`per_channel/02` produces one UE8M0 exponent byte per channel per token group,
and the consumer wants two token groups interleaved into each 16-bit word: byte
`2c` from group `2i`, byte `2c+1` from group `2i+1`.

On a vector unit that is an interleave instruction. ASC loads both byte rows and
calls `S.vintlv`:

*source:* `puzzles/asc/quant/answer/per_channel/02_round_packed_m.py:88-95`

<!-- snippet: interleave_asc -->
```python
a = S.vld(sf_rows_ub[0, col], dist="NORM_B8")
b = S.vld(sf_rows_ub[1, col], dist="NORM_B8")
# A uint8 register is 256 lanes, so `lo` already holds the
# interleave of the first 128 elements of each input --
# exactly the 256 output bytes for these 128 channels.
# `hi` interleaves the padding and is discarded.
lo, _ = S.vintlv(a, b)
S.vsts(packed_ub[col * PACK_FACTOR], lo, dist="NORM_B8")
```

PTO is the same shape, with the width spelled out:

*source:* `puzzles/pto/quant/answer/per_channel/02_round_packed_m.py:94-101`

<!-- snippet: interleave_pto -->
```python
a = V.vload(sf_rows_ub[0, col], size=BYTE_LANES * 2)
b = V.vload(sf_rows_ub[1, col], size=BYTE_LANES * 2)
# A uint8 register is 256 lanes, so `lo` already holds the
# interleave of the first 128 elements of each input --
# exactly the 256 output bytes for these 128 channels.
# `hi` interleaves the padding and is discarded.
lo, _ = V.vintlv(a, b, byte_mask)
V.vstore(lo, packed_ub[col * PACK_FACTOR])
```

There is a trap underneath both. A `uint8` vector register is **256 lanes**, so a
load of the 128 meaningful bytes reads 128 bytes past the end of the row. Both
SIMD tiers therefore have to pad the scratch buffer by a whole tile:

*source:* `puzzles/asc/quant/answer/per_channel/02_round_packed_m.py:41-41`

<!-- snippet: pad_asc -->
```python
sf_rows_ub = T.alloc_shared((PACK_FACTOR, hidden + BYTE_LANES), T.uint8)
```

*source:* `puzzles/pto/quant/answer/per_channel/02_round_packed_m.py:41-41`

<!-- snippet: pad_pto -->
```python
sf_rows_ub = T.alloc_shared((PACK_FACTOR, hidden + BYTE_LANES), T.uint8)
```

SIMT writes two scalar stores, and a scalar store cannot overrun — so the padding
disappears from the allocation as well as the instruction:

*source:* `puzzles/simt/quant/answer/per_channel/02_round_packed_m.py:74-76`

<!-- snippet: interleave_simt -->
```python
for c in T.Parallel(hidden):
    packed_ub[PACK_FACTOR * c] = sf_rows_ub[0, c]
    packed_ub[PACK_FACTOR * c + 1] = sf_rows_ub[1, c]
```

*source:* `puzzles/simt/quant/answer/per_channel/02_round_packed_m.py:44-44`

<!-- snippet: pad_simt -->
```python
sf_rows_ub = T.alloc_shared((PACK_FACTOR, hidden), T.uint8)
```

| | ASC | PTO | SIMT |
|---|---:|---:|---:|
| VF-region lines | 35 | 40 | **17** |
| simulated cycles | 849 | **811** | 5,964 |
| vs ASC | 1.00x | 0.96x | 7.02x |

Note PTO is *longer* than ASC here (40 lines against 35) while being marginally
faster. And SIMT, having removed both the instruction and the padding bug it
invites, costs **7.02x**.

### 3.2 A transpose: column-major scale output

`per_token/05` writes the scale array transposed, so a consumer can fetch a
tile's scales contiguously. The scales live one-per-group in vector lanes, and
**lane assignment inside a vector register is fixed**. ASC must therefore compute, per lane,
which source element that lane should receive — and then gather:

*source:* `puzzles/asc/quant/answer/per_token/05_col_major_sf.py:50-56`

<!-- snippet: transpose_asc -->
```python
lane = S.vci(0, T.int32)
token_of_lane = S.vand(T.reinterpret(lane, "uint32x64"),
                       S.vdup(token_block - 1, T.uint32))
group_of_lane = T.reinterpret(S.vshrs(lane, log2_block), "uint32x64")
S.vsts(idx_ub[0],
       S.vadd(S.vmuls(token_of_lane, SF_STRIDE), group_of_lane),
       dist="NORM_B32")
```

*source:* `puzzles/asc/quant/answer/per_token/05_col_major_sf.py:115-116`

<!-- snippet: transpose_asc_gather -->
```python
S.vsts(sf_out_ub[base, 0],
       S.vgather2(sf_dense_ub[0, base], idx), dist="NORM_B32")
```

PTO does the same thing and is slightly *more* verbose, because VMI has no
scalar-operand `vmul`: the stride constant must be broadcast into a vector
first:

*source:* `puzzles/pto/quant/answer/per_token/05_col_major_sf.py:51-60`

<!-- snippet: transpose_pto -->
```python
lane = V.vci(T.int32(0), size=LANES)
mask64 = V.create_mask(LANES, size=LANES)
token_of_lane = V.vand(V.vinterpret_cast(lane, "uint32"),
                       V.vbrc(T.uint32(token_block - 1), size=LANES))
group_of_lane = V.vinterpret_cast(
    V.vshrs(lane, log2_block, mask64), "uint32")
# No scalar-operand vmul in VMI: the stride must be a vector.
stride_v = V.vbrc(T.uint32(SF_STRIDE), size=LANES)
V.vstore(V.vadd(V.vmul(token_of_lane, stride_v, mask64),
                group_of_lane), idx_ub[0])
```

PTO gathers through it in turn, with the mask `vgather` requires:

*source:* `puzzles/pto/quant/answer/per_token/05_col_major_sf.py:114-115`

<!-- snippet: transpose_pto_gather -->
```python
V.vstore(V.vgather(sf_dense_ub[0, base], idx, gmask),
         sf_out_ub[base, 0])
```

In SIMT the transposed layout is simply the index the scale is stored at. The
output buffer is declared with its axes already swapped — groups down, tokens
across — while the input is still read one token at a time:

*source:* `puzzles/simt/quant/answer/per_token/05_col_major_sf.py:38-39`

<!-- snippet: transpose_simt_decl -->
```python
# Already transposed: one column per token of the current block.
sf_out_ub = T.alloc_shared((num_groups, token_block), T.float32)
```

and the store writes straight into the transposed slot. `sub` is the token
index, from the `T.serial(token_block)` loop outside the region; `g` is the
thread's group; everything above the last line is section 5.2's body unchanged:

*source:* `puzzles/simt/quant/answer/per_token/05_col_major_sf.py:56-65`

<!-- snippet: transpose_simt -->
```python
for g in T.Parallel(num_groups):
    amax = T.max(amax_l[g], E4M3_CLAMP_MIN)

    # ceil(log2(amax/448)) from the exponent field, then
    # both scales rebuilt by writing an exponent back.
    bits = T.reinterpret("int32", amax / E4M3_MAX)
    biased = ((bits - 1) >> 23) + 1
    inv_l[g] = T.reinterpret("float32", (254 - biased) << 23)
    # The whole column-major config: a different index.
    sf_out_ub[g, sub] = T.reinterpret("float32", biased << 23)
```

| | ASC | PTO | SIMT |
|---|---:|---:|---:|
| VF-region lines | 54 | 48 | **16** |
| simulated cycles | 5,477 | 5,448 | 10,897 |
| vs ASC | 1.00x | 0.99x | 1.99x |

**This is the largest source-size cut in the set.** SIMT's 16 lines against
ASC's 54 is the biggest reduction among the 22 variants, at 1.99x the cycles.
Until this variant's reduction was re-spelled (section 5.1) it was also the
sharpest *trade* in the deck, at 13.74x; what is left is the thread model
removing the per-lane index arithmetic and running a 128-lane unit one element
per thread.

One caveat on reading the ASC-vs-PTO numbers here. The 56-to-38 operation count
in PTO's favour is *not* the transpose — on the gather itself the two are a draw,
and PTO is marginally longer. The saving comes from the rest of the kernel, which
is `per_token/01` and `/02` machinery covered in section 5.

### The rule

> SIMT pays off exactly where the SIMD tiers had to move data between lanes:
> a group broadcast, an interleave, a transpose, a gather.

That is a narrow and useful claim. It also explains the cost: those are precisely
the cases where one thread handles one element, so the 128-lane unit runs at a
fraction of its width.

---

## 4. Where the operation already maps onto the hardware

The mirror image of section 3, and the question worth asking before reaching for
a thread model: when does the SIMD code have nothing to simplify?

### 4.1 When the data already lines up with the lanes

`per_channel/01` reduces along **M** — down the token axis — to get one scale per
channel. A channel *is* a lane, so there is no cross-lane step for anybody. ASC
keeps a register accumulator and maxes 32 loaded rows into it:

*source:* `puzzles/asc/quant/answer/per_channel/01_raw_32tokens.py:44-53`

<!-- snippet: chanreduce_asc -->
```python
acc = S.alloc_local((1,), T.float32)
# seeding with the clamp floor makes the clamp free
acc[0] = S.vdup(E4M3_CLAMP_MIN, T.float32)
for row in T.serial(group_tokens):
    v = S.vabs(S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"),
                      T.float32))
    # lane c holds the max over tokens seen so far
    acc[0] = S.vmax(acc[0], v)
S.vsts(sf_ub[col], S.vdiv(acc[0], qmax))
S.vsts(inv_ub[col], S.vdiv(qmax, acc[0]))
```

PTO is the same code with a mask operand:

*source:* `puzzles/pto/quant/answer/per_channel/01_raw_32tokens.py:46-55`

<!-- snippet: chanreduce_pto -->
```python
acc = V.alloc_local((1,), V.vreg(LANES, T.float32))
# seeding with the clamp floor makes the clamp free
acc[0] = V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES)
for row in T.serial(group_tokens):
    v = V.vabs(V.vcvt(V.vload(x_ub[row, col], size=LANES),
                      "float32"), mask)
    # lane c holds the max over tokens seen so far
    acc[0] = V.vmax(acc[0], v, mask)
V.vstore(V.vdiv(acc[0], qmax, mask), sf_ub[col])
V.vstore(V.vdiv(qmax, acc[0], mask), inv_ub[col])
```

SIMT is the mirror image — a channel is a *thread*, and the reduction is a serial
loop down 32 rows:

*source:* `puzzles/simt/quant/answer/per_channel/01_raw_32tokens.py:50-57`

<!-- snippet: chanreduce_simt -->
```python
for c in T.Parallel(hidden):
    amax = T.alloc_var(T.float32)
    amax = E4M3_CLAMP_MIN
    for row in T.serial(group_tokens):
        amax = T.max(amax,
                     T.abs(T.Cast("float32", x_ub[row, c])))
    sf_ub[c] = amax / E4M3_MAX
    inv_ub[c] = E4M3_MAX / amax
```

| | ASC | PTO | SIMT |
|---|---:|---:|---:|
| vector operations | 17 | 16 | — |
| VF-region lines | 21 | 21 | **12** |
| simulated cycles | 502 | **387** | 2,677 |
| vs ASC | 1.00x | 0.77x | 5.33x |

**17 operations against 16, and 21 source lines each.** This is the clearest
"no advantage" case in the set, and the reason is instructive: VMI's wins come
from expressing *segmented* behaviour and from width being an argument, and a
reduction that is already one instruction per vector offers neither. SIMT's 12
lines buy notation, not structure — the kernel has the same three stages and the
same memory traffic in all three styles.

Two honest qualifications. PTO is nonetheless **0.77x** in cycles, so "nothing to
buy" is a statement about source, not speed. And SIMT has no comparable
operation count at all: its arithmetic is plain operators, so the backend chooses
the instructions and there is nothing in the source to count. That is why the
operations row above is blank for SIMT throughout this report.

This is also the one family whose SIMT reduction is a serial loop rather than a
`T.reduce_absmax` call, and the reason is the axis. Section 5.1 shows the fold
being worth 8.73x where one thread per quant group leaves 124 of 128 threads
idle; here one thread per *channel* already uses all 128, so there is nothing to
recover — written as `T.reduce_absmax(..., dim=0)` this variant measures 2,530
cycles against 2,677, a 1.06x difference on a simulator, and one line longer.
The CUDA reference splits the same way: it folds `per_token` and `per_block` with
`T.reduce_absmax` / `T.reduce_max` and reduces `per_channel` with a per-thread
accumulator walked down the rows. On variants `02` and `04`, which write the
scale as a UE8M0 byte, the library call does not lower at all
(`doc/known-issues.md`).

### 4.2 When the value being placed is a scalar

`per_block/01` computes one scale per 32x32 tile. A single value has no layout,
so writing the scale array transposed — the same configuration that costs ASC a
lane-index vector and a gather in section 3.2 — is **free in all three styles**:
the index swap lives outside the `T.SimdVF` region entirely, on the final copy.

The contrast between those two variants is the cleanest statement of what the
thread model actually buys. Same configuration, same chip; what changes is
whether the data being relaid out sits in lanes.
---

## 5. Segmented reduction: PTO's largest measured advantage

### 5.1 The segmented reduction

`per_token/01` needs the maximum of each 32-channel group. ASC's vector reduction
produces one value per register, so four groups inside a 128-channel pair means
four masked reductions and four single-element stores:

*source:* `puzzles/asc/quant/answer/per_token/01_raw_fp32sf.py:59-70`

<!-- snippet: segreduce_asc -->
```python
a0 = S.vabs(S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32))
a1 = S.vabs(S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"),
                   T.float32))
# Each register holds two groups, so reduce half at a time.
S.vsts(amax_ub[group], S.vcmax(a0, mask_low), dist="ONEPT_B32")
S.vsts(amax_ub[group + 1], S.vcmax(a0, mask_high),
       dist="ONEPT_B32")
S.vsts(amax_ub[group + 2], S.vcmax(a1, mask_low),
       dist="ONEPT_B32")
S.vsts(amax_ub[group + 3], S.vcmax(a1, mask_high),
       dist="ONEPT_B32")
```

PTO says "reduce, in four segments" and the backend does the rest:

*source:* `puzzles/pto/quant/answer/per_token/01_raw_fp32sf.py:54-56`

<!-- snippet: segreduce_pto -->
```python
x = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
V.vstore(V.vcmax(V.vabs(x, mask), mask, group=groups_per_pair),
         amax_ub[group])
```

Fourteen vector operations become five. The matching broadcast on the apply pass
collapses the same way — four broadcast loads and two selects become grouped
slot `vload` + `vbrc`, fused by PTOAS.

SIMT names neither a lane nor a group. It gives one thread each channel and asks
the library for the fold, over a fragment whose *fast* axis is the quant group:

*source:* `puzzles/simt/quant/answer/per_token/01_raw_fp32sf.py:56-58`

<!-- snippet: segreduce_simt -->
```python
for g, j in T.Parallel(num_groups, group_size):
    x_l[g, j] = T.Cast("float32", x_ub[g * group_size + j])
T.reduce_absmax(x_l, amax_l, dim=1)
```

That is the same reshape the CUDA reference performs before its own
`T.reduce_absmax(..., dim=2)` (`per_token_cast_cuda.py` in TileKernels). The
clamp then needs a pass of its own, because a library fold cannot be seeded:

*source:* `puzzles/simt/quant/answer/per_token/01_raw_fp32sf.py:62-65`

<!-- snippet: segscale_simt -->
```python
for g in T.Parallel(num_groups):
    amax_l[g] = T.max(amax_l[g], E4M3_CLAMP_MIN)
    inv_l[g] = E4M3_MAX / amax_l[g]
    sf_l[g] = amax_l[g] / E4M3_MAX
```

| | ASC | PTO | SIMT |
|---|---:|---:|---:|
| vector operations | 39 | **20** | — |
| VF-region lines | 37 | 25 | **16** |
| simulated cycles | 6,176 | **4,576** | 8,204 |
| vs ASC | 1.00x | 0.74x | 1.33x |

**This is the one pattern where both axes agree.** PTO halves the operation
count *and* uses 0.74x the cycles, and SIMT is the shortest of the three at
1.33x.

#### What the other spelling cost: 8.73x

This variant is also where the SIMT tier's cycle numbers were decided, because
it shipped first with the reduction written the other way round — one thread per
*group*, walking its own 32 channels, with the clamp riding along as the
accumulator's seed:

*source:* `harness/probe/simt_reduce_max.py:109-113`

<!-- snippet: scan_simt -->
```python
for g in T.Parallel(num_groups):
    amax_l[g] = E4M3_CLAMP_MIN
    for j in T.serial(group_size):
        v = T.Cast("float32", x_ub[g * group_size + j])
        amax_l[g] = T.max(amax_l[g], T.abs(v))
```

With `group_size = 32` and `hidden = 128` there are four groups, so that
reduction occupies **4 of the 128 declared threads** — and issues its inner body
32 times over. `harness/probe/simt_reduce_max.py` builds both spellings of this
kernel, checks each against the same oracle, confirms they are byte-identical,
and measures each as the first device op of its own process:

| reduction spelling | VF_SIMT cycles | vs ASC's 6,176 | instructions |
|---|---:|---:|---:|
| one thread per group, inner `T.serial` | 71,606 | 11.59x | 36,676 |
| `T.reduce_absmax` — what this tier ships | **8,204** | **1.33x** | **4,036** |

**The width of a `T.Parallel` was worth 8.73x here**, and the tier was re-spelled
on the strength of it: every fold in `per_token` and `per_block` is now
`T.reduce_absmax`. `per_channel` still scans, for three measured reasons given in
section 11. Nothing about the arithmetic changed — the probe's two arms are
byte-identical — so this is the cost of one formulation, not of the thread model.

### 5.2 Width as an argument

`per_token/02` replaces the division with an exponent trick: round the scale up
to a power of two by reading and rewriting the float32 exponent field. ASC must
spell it inline, with the lane count baked into every type:

*source:* `puzzles/asc/quant/answer/per_token/02_round_sf.py:63-71`

<!-- snippet: exponent_asc -->
```python
one = S.vdup(1, T.uint32)
bits = T.reinterpret(S.vmuls(clamped, inv_qmax), "uint32x64")
# s = clamped/E4M3_MAX; this is ceil(log2(s)) + 127
biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
sf = T.reinterpret(S.vshls(biased, 23), "float32x64")
# 254 - biased == 127 - ceil_exp: the negated exponent
inv = T.reinterpret(
    S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
    "float32x64")
```

PTO writes the same arithmetic with the width as a parameter — so it can be
factored into a macro that serves 4, 64, 128 or 256 lanes:

*source:* `puzzles/pto/quant/answer/per_token/02_round_sf.py:35-44`

<!-- snippet: exponent_pto -->
```python
one = V.vbrc(T.uint32(1), size=lanes)
shift = V.vbrc(T.uint32(23), size=lanes)
bits = V.vinterpret_cast(
    V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=lanes), mask), "uint32")
# s = clamped/E4M3_MAX; this is ceil(log2(s)) + 127
biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
sf = V.vinterpret_cast(V.vshl(biased, shift), "float32")
# 254 - biased == 127 - ceil_exp: the negated exponent, still biased.
inv = V.vinterpret_cast(
    V.vshl(V.vsub(V.vbrc(T.uint32(254), size=lanes), biased), shift), "float32")
```

ASC structurally cannot write that function once, because `"uint32x64"` fixes the
width. This is VMI's deepest advantage and it does not show up as an operation
count — it shows up as reuse.

SIMT, having no width at all, writes the arithmetic as scalars. The fragment is
the whole `T.Parallel` body: its first six lines are the reduction of section
5.1, unchanged, and the scale arithmetic is the last four:

*source:* `puzzles/simt/quant/answer/per_token/02_round_sf.py:52-60`

<!-- snippet: exponent_simt -->
```python
for g in T.Parallel(num_groups):
    amax = T.max(amax_l[g], E4M3_CLAMP_MIN)

    bits = T.reinterpret("int32", amax / E4M3_MAX)
    biased = ((bits - 1) >> 23) + 1
    sf_l[g] = T.reinterpret("float32", biased << 23)
    inv_l[g] = T.reinterpret("float32", (254 - biased) << 23)
```

| | ASC | PTO | SIMT |
|---|---:|---:|---:|
| vector operations | 45 | **14** | — |
| VF-region lines | 43 | 20 | **18** |
| simulated cycles | 4,608 | 4,688 | 8,197 |
| vs ASC | 1.00x | 1.02x | 1.78x |

Two qualifications, both of which matter for a fair reading. The 45-to-14
operation count is the largest static gap in the whole set — and the cycles are a
**1.02x draw**. And part of PTO's apparent brevity here is bookkeeping: the macro
is defined *above* the measured region, so the three lines that call it are what
the count sees. The honest statement is that PTO's advantage on this pattern is
**reusability**, demonstrated by production calling that one macro at four
different widths, not fewer cycles.

---

## 6. Where PTO's explicitness costs source length

A comparison that only lists one frontend's wins is not usable. Three places
where VMI is the weaker surface:

### 6.1 Mandatory width and mask operands

`cast_back/02` decodes a UE8M0 exponent byte. ASC names the distribution mode on
the load and applies a per-lane shift vector:

*source:* `puzzles/asc/quant/answer/cast_back/02_packed_ue8m0.py:49-53`

<!-- snippet: ue8m0_asc -->
```python
packed = S.vld(sf_ub[strip], dist="BRC_B16")
bits = S.vand(S.vshl(T.reinterpret(packed, "uint32x64"),
                     sf_shift), exp_mask)
# An exponent field with a zero mantissa *is* 2^(e-127).
scale = T.reinterpret(bits, "float32x64")
```

PTO needs the slot count, expanded width, stride, group count and a mask spelled
out, and every constant lifted into a vector:

*source:* `puzzles/pto/quant/answer/cast_back/02_packed_ue8m0.py:52-56`

<!-- snippet: ue8m0_pto -->
```python
slot = V.vload(sf_ub[strip], size=1, stride=1, group=1)
packed = V.vbrc(slot, size=LANES * 2, group=1)
bits = V.vand(V.vshl(V.vinterpret_cast(packed, "uint32"),
                     sf_shift), exp_mask)
scale = V.vinterpret_cast(bits, "float32")
```

PTO writes 13 semantic operations against ASC's 12 and is four lines longer
(16 against 12). PTOAS fuses the slot load and broadcast, so the source count
does not imply an extra physical instruction. This is the variant where VMI buys
the least, and the mechanism is simply that `size=` and mask arguments make every
call wider.

### 6.2 A genuine missing capability

VMI has **no per-lane variable right shift**. ASC can build a vector of shift
amounts and apply `S.vshr(v, shifts)`; `V.vshrs` takes a scalar amount only, so
the production PTO `cast_back` computes both byte positions and selects between
them with `vcmp` + `vsel` where the ASC path did one shift.

SIMT expresses it without the question arising, because a per-element shift by a
per-element amount is ordinary scalar code:

*source:* `puzzles/simt/quant/answer/cast_back/02_packed_ue8m0.py:42-46`

<!-- snippet: ue8m0_simt -->
```python
for i in T.Parallel(hidden):
    g = i // group_size
    word = T.Cast("int32", sf_ub[g // PACK_FACTOR])
    byte = (word >> (8 * (g % PACK_FACTOR))) & 0xFF
    o_l[i] = q_l[i] * T.reinterpret("float32", byte << 23)
```

To be precise about scope: **no variant in this set exercises that regression** —
the kernels above use a left shift by a vector, which both frontends support. It
is reported here because it is the clearest case where ASC can express something
VMI cannot, and because a reader deciding between the two should know it exists.
That SIMT also writes `cast_back/02` in 9 lines against ASC's 12, at 3.20x — its
lowest ratio in the set — is the same point from the other side.

### 6.3 Verbosity with nothing to factor out

`per_channel/04` composes a bfloat16 reduction, a power-of-two scale and
M-packing. It is PTO's longest region in the set — **51 lines against ASC's 46**
— at a **1.00x** cycle draw. Where there is no segmented behaviour to collapse
and no width to parameterise, VMI's explicitness is pure cost.

---

## 7. Where SIMT loses outright

`per_token/04` quantizes to FP4 (e2m1). The hardware convert exists, and all
three frontends run on the same unit; what differs is whether the frontend has a
call that reaches it. ASC and PTO name it — `S.vcvt(v, T.float4_e2m1fn)`,
`V.vcvt(v, "float4_e2m1fn")` — so their only real work is pre-rounding to
bfloat16 with a sticky bit, to stop the hardware's second rounding from
double-rounding.

ASC pairs two registers to deinterleave them, then converts:

*source:* `puzzles/asc/quant/answer/per_token/04_fp32_in_fp4_out.py:73-78`

<!-- snippet: fp4_asc -->
```python
low, high = S.vdintlv(T.reinterpret(q0, "uint16x128"),
                      T.reinterpret(q1, "uint16x128"))
odd = S.vor(high, S.vmin(low, one_u16))
S.vsts(q_ub[col],
       S.vcvt(T.reinterpret(odd, "bfloat16x128"),
              T.float4_e2m1fn), dist="PK4_B32")
```

PTO's version is four operations, because `vunzip` splits a single value rather
than pairing two:

*source:* `puzzles/pto/quant/answer/per_token/04_fp32_in_fp4_out.py:31-33`

<!-- snippet: fp4_pto -->
```python
low, high = V.vunzip(x, "uint16")
one = V.vbrc(T.uint16(1), size=lanes)
return V.vinterpret_cast(V.vor(high, V.vmin(low, one)), "bfloat16")
```

It is a `@T.macro`, so the call site is where ASC's inline statements are:

*source:* `puzzles/pto/quant/answer/per_token/04_fp32_in_fp4_out.py:80-81`

<!-- snippet: fp4_pto_call -->
```python
odd = to_bf16_round_odd(q, PAIR)
V.vstore(V.vcvt(odd, "float4_e2m1fn", rounding="R"), q_ub[col])
```

**Nothing in the SIMT surface reaches that convert.** Inside a `T.SimtVF` region
the arithmetic is plain operators on scalars, and `T.copy` into an FP4 buffer
does not compile — `harness/probe/simt_limits.py` keeps that as a regression
test. The SIMT kernel therefore declares its output as `uint8` and encodes the
format by hand — sign, exponent, subnormal renormalization, guard/round/sticky,
saturation:

*source:* `puzzles/simt/quant/answer/per_token/04_fp32_in_fp4_out.py:61-66`

<!-- snippet: fp4_simt -->
```python
bits = T.reinterpret("int32", v)
# bit 3 of (bits >> 28) is bit 31 of bits: the sign nibble, without
# depending on whether >> sign-extends.
sign_nib = (bits >> 28) & 0x8
exps = (bits >> 23) & 0xFF
mant = bits & 0x7FFFFF
```

That is an extract; the full encoder is eighteen lines. It is bit-exact against
the PyTorch packer, verified before the kernel was ever run.

| | ASC | PTO | SIMT |
|---|---:|---:|---:|
| VF-region lines | 34 | 23 | 17 (+18 helper) |
| simulated cycles | 6,160 | **4,784** | 562,280 |
| vs ASC | 1.00x | 0.78x | **91.28x** |

`per_block/03` is the same loss for the same reason, at 10.18x — so this is a
pattern, not a one-off. It is also, after the reduction re-spelling of section
5.1, the only row left in the set above 11x.

### The 91.28x decomposes, and half of it is not SIMT's fault

Filing this under "hand-written encoders are expensive" would be too quick. The
gap splits into two independent factors, measured against this variant's sibling
SIMT kernels. Both columns come from the same place — the kernels' own traces,
counted by `tools/vf_trace_audit.py` over every vector-pipe instruction line:

| factor | measured |
|---|---|
| instructions issued | 243,130 against 4,036–5,828 |
| of those, pure control flow | 44,835 branches + 22,788 predicate sets |

The second factor is **branch divergence**. `T.if_then_else` on a data-dependent
per-element condition lowers to a real branch on this toolchain, not a predicated
select, and this is the only variant in the set whose trace carries a large
number of `SIMT_BRANCH` and `SIMT_START_DVG` markers. Respelling the encoder's
three conditionals as mask arithmetic — `m = -T.Cast("int32", cond)`, then
`(a & m) | (b & ~m)` — issues **37% fewer instructions and runs at 0.48x the
cycles**: 562,280 down to **269,370**, or 43.7x ASC rather than 91.3x.

Both spellings ship in the variant behind a `branch_free` flag and the harness
asserts they are byte-identical. On a GPU the distinction would not arise, since
a compiler predicates such a conditional into a select — so roughly a 2.09x slice
of this variant's gap is a **codegen gap, not a cost of the thread model**.

### The encoder is optional: 72.29x by leaving the region

The obvious question is whether a `T.Cast` to FP4 works inside a `SimtVF`
region, which would delete the encoder outright. It does not, and it cannot.
Nine spellings were tried — a scalar cast, a `T.vectorized(2)` pair, an fp4
fragment at 2, 4, 8 and 16 codes per thread, a `T.copy` from a float32
fragment, an explicit `T.Cast("float4_e2m1fnx2", ...)`, and a reinterpret of a
code as a byte — and all nine fail before codegen
(`harness/probe/simt_fp4_convert.py`). An fp4 element is **four bits**, so fp4
storage is retyped to `float4_e2m1fnx2` — one byte, two codes — and every store
needs `index_lanes * 2` lanes, where per-thread scalar code has one. Underneath,
the toolchain's only FP4 types are `vector_f4e2m1x2` and `float4_e2m1x2_t`, and
the conversion is `vcvt(vector_f4e2m1x2, vector_bf16, preg, ROUND_R, PART_P0)`:
a vector instruction over a vector-only type. **There is no scalar FP4 type to
convert into**, which is a property of the hardware rather than of tilelang.

What *does* work is leaving the region. A `T.SimtVF` region and a `T.SimdVF`
region can live in one kernel, so the thread model can keep the reduction, the
scales and the round-to-odd, and the vector model can be entered for the one
instruction it owns. It is a separate function in the variant file —
`compile_hybrid_kernel`, beside the tier's own `compile_kernel` — because the
two differ in their output stage and in their output dtype:

*source:* `puzzles/simt/quant/answer/per_token/04_fp32_in_fp4_out.py:251-256`

<!-- snippet: fp4_hwcvt_simt -->
```python
with T.SimdVF():
    for pair in T.serial(hidden // PAIR):
        S.vsts(q_ub[pair * PAIR],
               S.vcvt(S.vld(bf_ub[pair * PAIR]),
                      T.float4_e2m1fn),
               dist="PK4_B32")
```

| `per_token/04`, SIMT | cycles | vs ASC |
|---|---:|---:|
| the 18-line e2m1 encoder | 562,280 | 91.3x |
| three lines of `T.SimdVF` | **7,778** | **1.26x** |

**72.29x**, bit-exact against the torch packer and byte-identical to the other
two spellings, which the harness asserts. It is reached by
`launch(x, hw_convert=True)` and measured in the "Alternative spellings" section of
[`quant_vf_cycle_comparison.md`](../doc/quant_vf_cycle_comparison.md).

It is deliberately **not** the default, and this report's SIMT column does not
use it: a kernel that drops into `T.SimdVF` for its hardest instruction is no
longer an answer to "what does the thread model cost", which is the question the
tier exists to measure. It is a very good answer to a different question — what
a *mixed* kernel costs — and that question turns out to have a three-line
answer.

### The rule

> When the hardware has an instruction for the job, the frontend that can *name*
> it wins, and the margin is not close. All three run on the same vector unit;
> what differs is whether the frontend's surface has a call that reaches the
> instruction.
---

## 8. The inversion: static counts mispredict cycles

Sections 5 and 6 compared what gets written. This section is the warning label on
all of it.

`per_block` shows it most directly. PTO writes **more** vector operations than
ASC on four of the five variants, and runs fewer cycles on two of those four;
the fifth, `03`, is the one where PTO writes fewer operations:

| variant | ops ASC | ops PTO | cycles PTO/ASC |
|---|---:|---:|---:|
| `per_block/01_raw_32x32` | 21 | 22 | **0.85x** |
| `per_block/02_round_packed` | 27 | 30 | 1.02x |
| `per_block/04_col_major_tma` | 21 | 22 | **0.65x** |
| `per_block/05_split_compose` | 29 | 31 | 1.02x |

The mechanism is in the reduction loop. ASC reduces a 1024-value tile in
**16 steps of 64 lanes**:

*source:* `puzzles/asc/quant/answer/per_block/01_raw_32x32.py:52-56`

<!-- snippet: blockreduce_asc -->
```python
for chunk in T.serial(num_chunks):
    v = S.vabs(S.vcvt(S.vld(flat_x[chunk * LANES],
                            dist="UNPK_B16"), T.float32))
    S.vsts(run_ub[chunk], S.vcmax(v, mask_all),
           dist="ONEPT_B32")
```

PTO reaches 128 float32 lanes in one logical vector, so the same tile is
**8 steps of 128**:

*source:* `puzzles/pto/quant/answer/per_block/01_raw_32x32.py:53-56`

<!-- snippet: blockreduce_pto -->
```python
for chunk in T.serial(num_chunks):
    v = V.vabs(V.vcvt(V.vload(flat_x[chunk * LANES],
                              size=LANES), "float32"), mask)
    V.vstore(V.vcmax(v, mask, group=1), run_ub[chunk])
```

More operations written per step, half as many steps executed. A static count
cannot see the trip count, and here the trip count dominates.

SIMT has no trip count to compare, because it has no tile loop. A fragment's
shape is an index space, so the 32x32 tile is declared as a single row and the
whole thing folds in one call —

*source:* `puzzles/simt/quant/answer/per_block/01_raw_32x32.py:65-67`

<!-- snippet: blockreduce_simt -->
```python
for i, j in T.Parallel(bm, bk):
    flat_l[0, i * bk + j] = T.Cast("float32", x_ub[i, j])
T.reduce_absmax(flat_l, tile_l, dim=1)
```

— leaving one thread three scalar operations, not a loop:

*source:* `puzzles/simt/quant/answer/per_block/01_raw_32x32.py:69-72`

<!-- snippet: blockscale_simt -->
```python
for _z in T.Parallel(1):
    tile = T.max(tile_l[0], E4M3_CLAMP_MIN)
    sf_ub[0] = tile / E4M3_MAX
    inv_ub[0] = E4M3_MAX / tile
```

That is **2.45x**: 2,664 cycles against ASC's 1,086, from 13 source lines against 21.
The GPU shape — a partial per row into shared memory, then one thread folding the
32 partials while 127 wait — is what this variant shipped first, and it measured
19,276 cycles from 18 lines. The fold was most of the kernel.

`cast_back/05` shows the same independence in reverse: PTO's largest operation
reduction within `cast_back` (16 against 9, 18 lines against 13) and its **worst
cycle regression in the set, 1.13x**. Shorter and slower, in one variant. The
largest operation reduction overall is `per_token/02`, at 45 against 15, whose
cycle ratio is 1.02x.

The lesson for anyone using operation counts to choose a frontend: they measure
source size, which matters on its own, and they are not a performance model.

---

## 9. The complete measurement

All 22 variants. `ops` and `ln` are static counts inside the vector region;
`cyc` is simulated VF cycles. Bold marks PTO using meaningfully fewer cycles
than ASC.

| variant | ops A | ops P | ln A | ln P | ln S | cyc A | cyc P | cyc S | P/A | S/A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cast_back/01_e4m3_fp32sf` | 9 | 8 | 16 | 13 | 6 | 2,105 | 2,201 | 6,908 | 1.05x | 3.28x |
| `cast_back/02_packed_ue8m0` | 12 | 13 | 12 | 16 | 9 | 2,201 | 2,233 | 7,033 | 1.01x | 3.20x |
| `cast_back/03_block_sf` | 8 | 7 | 9 | 12 | 6 | 2,080 | 2,144 | 6,969 | 1.03x | 3.35x |
| `cast_back/04_per_channel_sf` | 6 | 6 | 6 | 7 | 6 | 2,016 | 2,048 | 6,983 | 1.02x | 3.46x |
| `cast_back/05_fp4_e2m1` | 16 | 9 | 18 | 13 | 13 | 2,296 | 2,584 | 10,152 | 1.13x | 4.42x |
| `cast_back/06_col_major_compose` | 14 | 15 | 21 | 25 | 16 | 2,272 | 2,528 | 9,275 | 1.11x | 4.08x |
| `per_token/01_raw_fp32sf` | 39 | 20 | 37 | 25 | 16 | 6,176 | 4,576 | 8,204 | **0.74x** | 1.33x |
| `per_token/02_round_sf` | 45 | 15 | 43 | 22 | 18 | 4,608 | 4,688 | 8,197 | 1.02x | 1.78x |
| `per_token/03_packed_ue8m0` | 44 | 16 | 37 | 23 | 16 | 4,608 | 4,672 | 9,893 | 1.01x | 2.15x |
| `per_token/04_fp32_in_fp4_out` | 37 | 18 | 34 | 24 | 17 | 6,160 | 4,784 | 562,280 | **0.78x** | 91.28x |
| `per_token/05_col_major_sf` | 56 | 39 | 54 | 50 | 16 | 5,477 | 5,448 | 10,897 | 0.99x | 1.99x |
| `per_token/06_split_requant` | 39 | 32 | 57 | 50 | 33 | 6,864 | 4,576 | 8,204 | **0.67x** | 1.20x |
| `per_token/07_bf16_fast_compose` | 31 | 18 | 39 | 26 | 16 | 4,864 | 5,136 | 8,502 | 1.06x | 1.75x |
| `per_block/01_raw_32x32` | 21 | 22 | 21 | 25 | 13 | 1,086 | 924 | 2,664 | **0.85x** | 2.45x |
| `per_block/02_round_packed` | 27 | 30 | 28 | 33 | 15 | 918 | 936 | 3,130 | 1.02x | 3.41x |
| `per_block/03_fp4_e2m1` | 28 | 26 | 29 | 29 | 14 | 1,625 | 1,040 | 16,536 | **0.64x** | 10.18x |
| `per_block/04_col_major_tma` | 21 | 22 | 21 | 25 | 13 | 1,414 | 924 | 3,044 | **0.65x** | 2.15x |
| `per_block/05_split_compose` | 29 | 31 | 33 | 37 | 24 | 918 | 940 | 3,130 | 1.02x | 3.41x |
| `per_channel/01_raw_32tokens` | 17 | 16 | 21 | 21 | 12 | 502 | 387 | 2,677 | **0.77x** | 5.33x |
| `per_channel/02_round_packed_m` | 28 | 29 | 35 | 40 | 17 | 849 | 811 | 5,964 | **0.96x** | 7.02x |
| `per_channel/03_requant_bf16` | 23 | 20 | 29 | 38 | 14 | 930 | 721 | 4,073 | **0.78x** | 4.38x |
| `per_channel/04_compose` | 32 | 33 | 46 | 51 | 17 | 738 | 736 | 5,964 | 1.00x | 8.08x |
| **total** | **582** | **445** | **646** | **605** | **327** | **60,707** | **55,037** | **710,679** | | |

Aggregates:

| | PTO vs ASC | SIMT vs ASC |
|---|---|---|
| vector operations | 76% | — (no comparable count) |
| source lines | 94% | **51%** |
| simulated cycles | **0.92x mean** (0.64x–1.13x) | 7.71x mean (1.20x–91.28x) |

PTO is faster than ASC on **9 of 22** variants.

Excluding `per_token/04`, whose hand-written FP4 encoder is a special case
(section 7), SIMT's range is **1.20x–10.18x, mean 3.73x**. That is the
figure to carry away for the thread model as such; the 91.28x row measures an
instruction the SIMT surface cannot name, not a programming style.

### Source size and cycle count, side by side

![VF-region source lines per variant, the three frontends side by side](plots/lines.png)

- **SIMT** writes **51%** of ASC's lines and costs 3.73x the cycles outside the
  FP4 row. The win is concentrated in lane-placement work; what is left of the
  cost is format work the surface cannot name and per-element issue — the
  `cast_back` family, which has no reduction to win, is the floor at 3.20x.
- **PTO** writes **76%** of ASC's operations, **94%** of its lines, and
  costs **0.92x** of its cycles. On this evidence it is not a trade at all:
  it is shorter *and*, on average, cheaper.
- **ASC** remains the only frontend that can express every instruction the
  machine has (section 6.2). On cycles the ASC-vs-PTO split is **9 variants to
  PTO, 6 to ASC, 7 within 2%** — so "ASC is fastest because it is closest to the
  machine" is not what the measurement says.
---

## 10. Method, and why the numbers can be trusted

A cycle comparison is only as good as its provenance, so this section states the
method and the two failure modes that were actually encountered.

**Measurement.** Each kernel runs under `msprof op simulator`
(SoC `Ascend950PR_9599`). Simulated VF cycles are the sum of the `cycles` column
of `core0.veccore0_instr_exe.csv` over the rows whose `instr` names a VF launch —
`VF` for a SIMD kernel, `VF_SIMT` for a SIMT one. `cycles` is already the total
for its row and is never scaled by `call_count`.

A detail worth recording for anyone repeating this: the filter must key on the
**`instr`** column, not `pipe`. Every real VF row, SIMD and SIMT alike, is on
pipe `PUSHQ`, and `MOVEMASK` / `PUSH_PB` share that lane.

**Correctness first.** Every number above comes from a kernel that passed the
same check as its siblings — compared against a standalone PyTorch oracle, with
an assertion that the result came back from the device rather than being computed
on the host. No cycle count is reported for a kernel that computed the wrong
answer.

**Like-for-like.** A cheaper kernel and a smaller kernel look identical in a
cycle total, so VF launch counts were compared across tiers on every variant.
They match everywhere except `per_token/05`, where ASC and PTO issue 34 against
SIMT's 32 — the two extra being the one-time gather-index setup that SIMT does
not need.

**Code fragments.** No fragment in this report is typed by hand. Each is cut
from a kernel the harness passes, by `tools/make_report_snippets.py`, which
locates it by an anchor line rather than a line number and re-resolves every cut
on each run; `tools/check_report_numbers.py` fails if an embedded block or the
`*source:*` range above it has drifted. Within a variant the fragments quoted
side by side cover the same step of the algorithm, so they can be read against
each other: where a SIMD fragment's enclosing `for ... in T.serial(...)` tile
loop is left out, the SIMT fragment's `T.Parallel` header is kept, because that
header is what replaces it. The two exceptions are stated where they appear —
the e2m1 encoder in section 7, quoted as 6 lines of 18, and PTO's `@T.macro`
bodies, which are shown with their call sites.

### Two failure modes that were caught

**The profiler measured the wrong kernel.** `msprof op simulator` profiles a
*single operator* — in practice the first the process launches. Two variants
allocate a placeholder input with `torch.zeros(..., device=...)` inside their
host wrapper, which on the NPU is a fill *kernel* that runs first. All three
tiers had therefore measured the same aten fill. It was caught because the three
reported **identical** cycle counts, which three different implementations cannot
do. Those variants are now measured through a driver that makes the kernel the
first operator, and the extraction tool refuses any variant whose tiers report
identical cycles.

**Verifying the trace is the kernel.** Because of the above, every trace is now
audited against the simulator's instruction log, which names each instruction
that issued, vector pipes included. The instruction mix is checkable by hand: for
`cast_back/01` on ASC at 64 lanes — two strips per token — the log shows exactly
6 `RV_VLDI`, 4 `RV_VCVT_F2F`, 2 `RV_VSEL`, 2 `RV_VMUL` and 2 `RV_VSTI` per
launch, which is the source's per-strip structure doubled. All 66 vector-tier
traces pass; the `torch.zeros` fill, by contrast, issues one `RV_VDUPS` and two
stores and no multiply at all.

---

## 11. Caveats

Stated plainly, because they bound every conclusion above.

1. **These are simulator cycles, not silicon.** No hardware-fidelity claim is
   made.
2. **One shape per variant.** Mostly `M=32, K=128`; `per_token/07` runs at
   `K=256` and the two M-packed `per_channel` variants at `M=64`. Ratios are
   within-variant and valid; cycle counts are not comparable between rows.
3. **Single core, no scheduling.** `T.Kernel(1)` throughout, so nothing here
   speaks to multi-core behaviour, double buffering or cache hints.
4. **The SIMT reductions are spelled for speed; nothing else is.** Every fold in
   `per_token` and `per_block` is `T.reduce_absmax`, after section 5.1 measured
   the thread-per-group scan at **8.73x** on `per_token/01`. `per_channel` still
   scans, for three measured reasons: its fold runs down the token axis, where
   one thread per channel already uses all 128 threads (1.06x, measured); the
   CUDA reference scans the same axis the same way; and on its two byte-packing
   variants the library call does not lower at all (`doc/known-issues.md`). The
   format work — the e2m1 encoders, the UE8M0 packs — and the `cast_back` family
   are untuned. A separate SIMT-versus-SIMD suite on the same simulator, sweeping
   several SIMT strategies per kernel, found ratios from **0.33x to 53x** — that
   is, SIMT *winning* by 3x on some grouped reductions. In one case the strategy
   choice alone moved an identical kernel by a factor of two. **Nothing here
   should be read as "SIMT is slow" in general.**
5. **Static line counts exclude helpers outside the vector region**, in every
   frontend. That understates SIMT on the two FP4 variants by about 18 lines each
   — the hand-written encoder lives in a module-level helper.
6. **The group sizes here are small.** Every reduction uses `group_size = 32`,
   which is the case that leaves the most SIMT threads idle. A kernel whose
   natural parallelism matches the thread count would not look like this.

---

## 12. Conclusions

**Where SIMT reduces source size.** Only where the SIMD frontends
had to move data *between lanes*. The three cases in this set are a group
broadcast (section 2), an interleave (3.1) and a transpose (3.2), and they are
also where SIMD code is at its least readable — `S.vci` plus `S.vgather2` to
express "store it transposed", or an interleave instruction plus a padded scratch
row to express "write two bytes". There SIMT is a third of the code and removes
a class of index error. Elsewhere its shorter source expresses the same stages
and the same memory traffic, at 1.20x to 10.18x the cycles — under 3x on 8 of
the 22 variants and over 6x on 4. The reductions got there by being re-spelled:
`T.reduce_absmax` in place of a thread-per-group scan moved `per_token/01` from
11.59x to 1.33x and `per_block/01` from 17.75x to 2.45x without changing a
result (section 5.1). What is left of the gap is format work the SIMT surface
cannot name, and per-element issue on the kernels that have no reduction to win.

**Where the SIMD frontends have nothing to simplify.** When the operation maps onto
the hardware as written. If the data already lines up with the lanes
(`per_channel/01`: a channel *is* a lane), or the value being placed is a scalar
(`per_block/01`: a tile scale has no layout), then ASC is already saying exactly
what it means, VMI has nothing to factor out, and SIMT has no lane placement to
remove. Those variants come out 17 operations against 16, and 21 lines against
21. Reaching for a thread model there buys brevity and nothing else — though
`per_block/01` is also where SIMT's one structural advantage shows up, since a
fragment's shape lets the whole tile fold in a single call.

**What each frontend is for.**

| | choose it when |
|---|---|
| **SIMT** | getting a correct kernel quickly matters more than cycles; or the kernel is dominated by lane placement — packing, transposes, gathers — where it is both shorter and less error-prone |
| **PTO/VMI** | as the default for production vector code. On this evidence it is shorter than ASC (76% of the operations) *and* no slower (0.92x the cycles), and `size=`-parameterised helpers are reusable across widths in a way ASC structurally cannot be |
| **ASC** | an instruction VMI cannot express is needed — a per-lane variable right shift being the clearest case — or a specific distribution mode is the point |

**The methodological conclusion, which generalises past this chip.** Operation
counts and line counts measure source size. They were a poor predictor of
cycles here: PTO writes 76% of ASC's operations and uses 92% of its cycles, wins
on 9 variants of 22, and on four of those *writes more operations than ASC*. The
`per_block` family shows why — a wider logical vector means more operations
written per step and half as many steps executed, and a static count cannot see
a trip count. Any frontend comparison that reports only source size is reporting
half the answer.

---

## Reproducing this

```bash
python -m harness.check                        # all 88 implementations
python tools/vf_lines.py                       # static counts + cycles per tier
python tools/vf_cycles.py --all --report       # re-measure under the simulator
python tools/vf_trace_audit.py                 # prove each trace is the kernel
python tools/make_report_snippets.py --check   # every snippet matches its source
python tools/check_report_numbers.py           # every number here re-derived
```

The per-variant walkthroughs behind this report are in `doc/quant/`, the
frontend overviews in `doc/tiers/`, the measured cycle table in
`doc/quant_vf_cycle_comparison.md`, and the raw data in
`perf/quant_vf_cycles.json`.
