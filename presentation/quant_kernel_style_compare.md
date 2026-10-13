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

| frontend | what you name | the unit you think in |
|---|---|---|
| **ASC** (`T.simd`, imported `S`) | the physical instruction | a 64-lane vector register |
| **PTO/VMI** (`T.vmi`, imported `V`) | the intent; width and segment count are *arguments* | a logical vector of any width |
| **SIMT** (`T.SimtVF` + `T.Parallel`) | one element | a thread |

The comparison rests on 22 kernel variants across four quantization kernels
(`cast_back`, `per_token`, `per_block`, `per_channel`), each implemented in all
three styles plus a PyTorch reference — 88 implementations, all passing the same
correctness checks against the same oracle. This report shows eight of the 22,
chosen because each isolates one mechanism. The rest corroborate.

**Out of scope**, deliberately: multi-core scheduling, double-buffered Unified
Buffer, and cache hints. Every kernel here is single-core `T.Kernel(1)`, so the
comparison is of the vector body and nothing else.

### The result that shapes the rest of this report

It is tempting to assume that the frontend which writes fewer operations
produces the faster kernel. Across these 22 variants that assumption is wrong
often enough to be the most important finding:

| PTO relative to ASC | |
|---|---|
| vector operations written | **74%** |
| source lines written | **89%** |
| simulated VF cycles | **92% mean**, range 0.64x–1.13x |

PTO uses fewer cycles than ASC on **9 of 22** variants — and on four of those it
*writes more operations than ASC*. So this report is organised around two axes
that disagree: how much you write, and what it costs. Each pattern below is
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

*source:* `puzzles/asc/quant/answer/cast_back/01_e4m3_fp32sf.py:66-68`

<!-- snippet: broadcast_asc -->
```python
lo = S.vld(sf_ub[group], dist="BRC_B32")
hi = S.vld(sf_ub[group + 1], dist="BRC_B32")
scale = S.vsel(lo, hi, mask_low)
```

PTO states the shape instead of constructing it. One load, two segments, each
broadcast from a consecutive scalar:

*source:* `puzzles/pto/quant/answer/cast_back/01_e4m3_fp32sf.py:60-61`

<!-- snippet: broadcast_pto -->
```python
scale = V.vload(sf_ub[group], size=LANES, stride=1,
                dist_mode="brc", group=groups_per_strip)
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

*source:* `puzzles/pto/quant/answer/cast_back/01_e4m3_fp32sf.py:70-73`

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

## 3. Where SIMT genuinely simplifies: irregular data *placement*

If SIMT has a thesis, it is this section. Both cases below are ones where the
SIMD tiers must move data **between lanes**, and a vector register's lane
assignment is fixed.

### 3.1 An interleave: packing scales along M

`per_channel/02` produces one UE8M0 exponent byte per channel per token group,
and the consumer wants two token groups interleaved into each 16-bit word: byte
`2c` from group `2i`, byte `2c+1` from group `2i+1`.

On a vector unit that is an interleave instruction. ASC loads both byte rows and
calls `S.vintlv`:

*source:* `puzzles/asc/quant/answer/per_channel/02_round_packed_m.py:88-91`

<!-- snippet: interleave_asc -->
```python
a = S.vld(sf_rows_ub[0, col], dist="NORM_B8")
b = S.vld(sf_rows_ub[1, col], dist="NORM_B8")
# A uint8 register is 256 lanes, so `lo` already holds the
# interleave of the first 128 elements of each input --
```

PTO is the same shape, with the width spelled out:

*source:* `puzzles/pto/quant/answer/per_channel/02_round_packed_m.py:94-97`

<!-- snippet: interleave_pto -->
```python
a = V.vload(sf_rows_ub[0, col], size=BYTE_LANES * 2)
b = V.vload(sf_rows_ub[1, col], size=BYTE_LANES * 2)
# A uint8 register is 256 lanes, so `lo` already holds the
# interleave of the first 128 elements of each input --
```

There is a trap underneath both. A `uint8` vector register is **256 lanes**, so a
load of the 128 meaningful bytes reads 128 bytes past the end of the row. Both
SIMD tiers therefore have to pad the scratch buffer by a whole tile:

*source:* `puzzles/asc/quant/answer/per_channel/02_round_packed_m.py:41-41`

<!-- snippet: pad_asc -->
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
**you cannot restride a register**. ASC must therefore compute, for every lane,
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

*source:* `puzzles/pto/quant/answer/per_token/05_col_major_sf.py:57-59`

<!-- snippet: transpose_pto -->
```python
stride_v = V.vbrc(T.uint32(SF_STRIDE), size=LANES)
V.vstore(V.vadd(V.vmul(token_of_lane, stride_v, mask64),
                group_of_lane), idx_ub[0])
```

In SIMT the transposed layout is simply the index the scale is stored at. There
is no index vector, no gather, and no block of scales held back to be permuted:

*source:* `puzzles/simt/quant/answer/per_token/05_col_major_sf.py:63-63`

<!-- snippet: transpose_simt -->
```python
sf_out_ub[g, sub] = T.reinterpret("float32", biased << 23)
```

| | ASC | PTO | SIMT |
|---|---:|---:|---:|
| VF-region lines | 54 | 48 | **15** |
| simulated cycles | 5,477 | 5,448 | 75,281 |
| vs ASC | 1.00x | 0.99x | 13.74x |

**This is the sharpest single trade in the set.** SIMT's 15 lines against ASC's
54 is the largest source-length win anywhere in the 22 variants, and 13.74x is
among the worst cycle ratios. The thread model removes the lane-placement problem
and leaves most of the vector unit idle doing it.

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

## 4. Where PTO's SIMD style is already clean enough

The mirror image of section 3, and the question worth asking before reaching for
a thread model: when does the SIMD code have nothing to simplify?

### 4.1 When the data already lines up with the lanes

`per_channel/01` reduces along **M** — down the token axis — to get one scale per
channel. A channel *is* a lane, so there is no cross-lane step for anybody. ASC
keeps a register accumulator and maxes 32 loaded rows into it:

*source:* `puzzles/asc/quant/answer/per_channel/01_raw_32tokens.py:44-49`

<!-- snippet: chanreduce_asc -->
```python
acc = S.alloc_local((1,), T.float32)
# seeding with the clamp floor makes the clamp free
acc[0] = S.vdup(E4M3_CLAMP_MIN, T.float32)
for row in T.serial(group_tokens):
    v = S.vabs(S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"),
                      T.float32))
```

PTO is the same code with a mask operand:

*source:* `puzzles/pto/quant/answer/per_channel/01_raw_32tokens.py:46-51`

<!-- snippet: chanreduce_pto -->
```python
acc = V.alloc_local((1,), V.vreg(LANES, T.float32))
# seeding with the clamp floor makes the clamp free
acc[0] = V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES)
for row in T.serial(group_tokens):
    v = V.vabs(V.vcvt(V.vload(x_ub[row, col], size=LANES),
                      "float32"), mask)
```

SIMT is the mirror image — a channel is a *thread*, and the reduction is a serial
loop down 32 rows:

*source:* `puzzles/simt/quant/answer/per_channel/01_raw_32tokens.py:50-56`

<!-- snippet: chanreduce_simt -->
```python
for c in T.Parallel(hidden):
    amax = T.alloc_var(T.float32)
    amax = E4M3_CLAMP_MIN
    for row in T.serial(group_tokens):
        amax = T.max(amax,
                     T.abs(T.Cast("float32", x_ub[row, c])))
    sf_ub[c] = amax / E4M3_MAX
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

### 4.2 When the value being placed is a scalar

`per_block/01` computes one scale per 32x32 tile. A single value has no layout,
so writing the scale array transposed — the same configuration that costs ASC a
lane-index vector and a gather in section 3.2 — is **free in all three styles**:
the index swap lives outside the vector body entirely, on the final copy.

The contrast between those two variants is the cleanest statement of what the
thread model actually buys. Same configuration, same chip; what changes is
whether the data being relaid out sits in lanes.
---

## 5. Where PTO's logical style is the real win

### 5.1 The segmented reduction

`per_token/01` needs the maximum of each 32-channel group. ASC's vector reduction
produces one value per register, so four groups inside a 128-channel pair means
four masked reductions and four single-element stores:

*source:* `puzzles/asc/quant/answer/per_token/01_raw_fp32sf.py:63-69`

<!-- snippet: segreduce_asc -->
```python
S.vsts(amax_ub[group], S.vcmax(a0, mask_low), dist="ONEPT_B32")
S.vsts(amax_ub[group + 1], S.vcmax(a0, mask_high),
       dist="ONEPT_B32")
S.vsts(amax_ub[group + 2], S.vcmax(a1, mask_low),
       dist="ONEPT_B32")
S.vsts(amax_ub[group + 3], S.vcmax(a1, mask_high),
       dist="ONEPT_B32")
```

PTO says "reduce, in four segments" and the backend does the rest:

*source:* `puzzles/pto/quant/answer/per_token/01_raw_fp32sf.py:55-56`

<!-- snippet: segreduce_pto -->
```python
V.vstore(V.vcmax(V.vabs(x, mask), mask, group=groups_per_pair),
         amax_ub[group])
```

Eight operations become three. The matching broadcast on the apply pass collapses
the same way — four broadcast loads and two selects become one `vload` with
`dist_mode="brc", group=4`.

SIMT puts one thread on each group and scans serially:

*source:* `puzzles/simt/quant/answer/per_token/01_raw_fp32sf.py:53-59`

<!-- snippet: segreduce_simt -->
```python
for g in T.Parallel(num_groups):
    amax_l[g] = E4M3_CLAMP_MIN
    for j in T.serial(group_size):
        v = T.Cast("float32", x_ub[g * group_size + j])
        amax_l[g] = T.max(amax_l[g], T.abs(v))
    inv_l[g] = E4M3_MAX / amax_l[g]
    sf_l[g] = amax_l[g] / E4M3_MAX
```

| | ASC | PTO | SIMT |
|---|---:|---:|---:|
| vector operations | 39 | **19** | — |
| VF-region lines | 37 | 24 | **15** |
| simulated cycles | 6,176 | **4,576** | 71,606 |
| vs ASC | 1.00x | 0.74x | 11.59x |

**This is the one pattern where both axes agree.** PTO halves the operation
count *and* uses 0.74x the cycles. It is also where SIMT's cost first becomes an
order of magnitude, and the reason is visible in the code above: with
`group_size = 32` and `hidden = 128` there are four groups, so the reduction runs
**4 threads out of 128**.

### 5.2 Width as an argument

`per_token/02` replaces the division with an exponent trick: round the scale up
to a power of two by reading and rewriting the float32 exponent field. ASC must
spell it inline, with the lane count baked into every type:

*source:* `puzzles/asc/quant/answer/per_token/02_round_sf.py:63-69`

<!-- snippet: exponent_asc -->
```python
one = S.vdup(1, T.uint32)
bits = T.reinterpret(S.vmuls(clamped, inv_qmax), "uint32x64")
# ((bits - 1) >> 23) + 1  ==  ceil(log2(v)) + 127
biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
sf = T.reinterpret(S.vshls(biased, 23), "float32x64")
# 254 - biased == 127 - ceil_exp: the negated exponent
inv = T.reinterpret(
```

PTO writes the same arithmetic with the width as a parameter — so it can be
factored into a macro that serves 4, 64, 128 or 256 lanes:

*source:* `puzzles/pto/quant/answer/per_token/02_round_sf.py:35-41`

<!-- snippet: exponent_pto -->
```python
one = V.vbrc(T.uint32(1), size=lanes)
shift = V.vbrc(T.uint32(23), size=lanes)
bits = V.vinterpret_cast(
    V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=lanes), mask), "uint32")
# ((bits - 1) >> 23) + 1  ==  ceil(log2(v)) + 127
biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
sf = V.vinterpret_cast(V.vshl(biased, shift), "float32")
```

ASC structurally cannot write that function once, because `"uint32x64"` fixes the
width. This is VMI's deepest advantage and it does not show up as an operation
count — it shows up as reuse.

SIMT, having no width at all, writes the arithmetic as scalars:

*source:* `puzzles/simt/quant/answer/per_token/02_round_sf.py:54-57`

<!-- snippet: exponent_simt -->
```python
bits = T.reinterpret("int32", amax / E4M3_MAX)
biased = ((bits - 1) >> 23) + 1
sf_l[g] = T.reinterpret("float32", biased << 23)
inv_l[g] = T.reinterpret("float32", (254 - biased) << 23)
```

| | ASC | PTO | SIMT |
|---|---:|---:|---:|
| vector operations | 45 | **14** | — |
| VF-region lines | 43 | 20 | **17** |
| simulated cycles | 4,608 | 4,688 | 71,008 |
| vs ASC | 1.00x | 1.02x | 15.41x |

Two qualifications, both of which matter for a fair reading. The 45-to-14
operation count is the largest static gap in the whole set — and the cycles are a
**1.02x draw**. And part of PTO's apparent brevity here is bookkeeping: the macro
is defined *above* the measured region, so the three lines that call it are what
the count sees. The honest statement is that PTO's advantage on this pattern is
**reusability**, demonstrated by production calling that one macro at four
different widths, not fewer cycles.

---

## 6. Where PTO costs you

A comparison that only lists one frontend's wins is not usable. Three places
where VMI is the weaker surface:

### 6.1 Mandatory width and mask operands

`cast_back/02` decodes a UE8M0 exponent byte. ASC names the distribution mode on
the load and applies a per-lane shift vector:

*source:* `puzzles/asc/quant/answer/cast_back/02_packed_ue8m0.py:49-52`

<!-- snippet: ue8m0_asc -->
```python
packed = S.vld(sf_ub[strip], dist="BRC_B16")
bits = S.vand(S.vshl(T.reinterpret(packed, "uint32x64"),
                     sf_shift), exp_mask)
# An exponent field with a zero mantissa *is* 2^(e-127).
```

PTO needs the width, the stride, the group count and a mask spelled out, and
every constant lifted into a vector:

*source:* `puzzles/pto/quant/answer/cast_back/02_packed_ue8m0.py:52-56`

<!-- snippet: ue8m0_pto -->
```python
packed = V.vload(sf_ub[strip], size=LANES * 2, stride=1,
                 dist_mode="brc", group=1)
bits = V.vand(V.vshl(V.vinterpret_cast(packed, "uint32"),
                     sf_shift), exp_mask)
scale = V.vinterpret_cast(bits, "float32")
```

Operations tie at 12 each; **PTO is four lines longer** (16 against 12). This is
the variant where VMI buys the least, and the mechanism is simply that `size=`
and mask arguments make every call wider.

### 6.2 A genuine missing capability

VMI has **no per-lane variable right shift**. ASC can build a vector of shift
amounts and apply `S.vshr(v, shifts)`; `V.vshrs` takes a scalar amount only, so
the production PTO `cast_back` computes both byte positions and selects between
them with `vcmp` + `vsel` where the ASC path did one shift.

To be precise about scope: **no variant in this set exercises that regression** —
the kernels above use a left shift by a vector, which both frontends support. It
is reported here because it is the clearest case where ASC can express something
VMI cannot, and because a reader deciding between the two should know it exists.

### 6.3 Verbosity with nothing to factor out

`per_channel/04` composes a bfloat16 reduction, a power-of-two scale and
M-packing. It is PTO's longest region in the set — **51 lines against ASC's 46**
— at a **1.00x** cycle draw. Where there is no segmented behaviour to collapse
and no width to parameterise, VMI's explicitness is pure cost.

---

## 7. Where SIMT loses outright

`per_token/04` quantizes to FP4 (e2m1). ASC and PTO both own a hardware
`float32 -> e2m1` convert; their only real work is pre-rounding to bfloat16 with
a sticky bit so the hardware's second rounding cannot double-round.

ASC pairs two registers to deinterleave them, then converts:

*source:* `puzzles/asc/quant/answer/per_token/04_fp32_in_fp4_out.py:73-77`

<!-- snippet: fp4_asc -->
```python
low, high = S.vdintlv(T.reinterpret(q0, "uint16x128"),
                      T.reinterpret(q1, "uint16x128"))
odd = S.vor(high, S.vmin(low, one_u16))
S.vsts(q_ub[col],
       S.vcvt(T.reinterpret(odd, "bfloat16x128"),
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

**No FP4 store is reachable from a `SimtVF` region.** The SIMT kernel therefore
declares its output as `uint8` and encodes the format by hand — sign, exponent,
subnormal renormalization, guard/round/sticky, saturation:

*source:* `puzzles/simt/quant/answer/per_token/04_fp32_in_fp4_out.py:62-67`

<!-- snippet: fp4_simt -->
```python
sh = T.min(126 - exps, 31)
mant_pre = 0x400000 | (mant >> 1)         # the implicit 1, then make room
low = (1 << sh) - 1                       # the bits about to be shifted out

if branch_free:
    # -(cond) is 0 or all-ones. Both arms are always evaluated, which is
```

That is an extract; the full encoder is eighteen lines. It is bit-exact against
the PyTorch packer, verified before the kernel was ever run.

| | ASC | PTO | SIMT |
|---|---:|---:|---:|
| VF-region lines | 34 | 23 | 15 (+18 helper) |
| simulated cycles | 6,160 | **4,784** | 613,632 |
| vs ASC | 1.00x | 0.78x | **99.62x** |

`per_block/03` is the same loss for the same reason, at 16.77x — so this is a
pattern, not a one-off.

### The 99.62x decomposes, and half of it is not SIMT's fault

Filing this under "hand-written encoders are expensive" would be too quick. The
gap splits into two independent factors, measured against this variant's sibling
SIMT kernels:

| factor | measured |
|---|---|
| instructions issued | 124,294 against ~37,000 |
| cycles per instruction | 4.94 against 1.90–2.01 |

The second factor is **branch divergence**. `T.if_then_else` on a data-dependent
per-element condition lowers to a real branch on this toolchain, not a predicated
select, and this is the only variant in the set whose trace carries a large
number of `SIMT_START_DVG` markers. Respelling the encoder's three conditionals
as mask arithmetic — `m = -T.Cast("int32", cond)`, then `(a & m) | (b & ~m)` —
issues **45% more instructions and runs 1.78x fewer cycles**: 613,632 down to
**345,189**, or 56.0x ASC rather than 99.6x.

Both spellings ship in the variant behind a `branch_free` flag and the harness
asserts they are byte-identical. On a GPU the distinction would not arise, since
a compiler predicates such a conditional into a select — so roughly a 1.78x slice
of this variant's gap is a **codegen gap, not a cost of the thread model**.

### The rule

> When the hardware has an instruction for the job, the frontend that can *name*
> it wins, and the margin is not close.
---

## 8. The inversion: static counts mispredict cycles

Sections 5 and 6 compared what gets written. This section is the warning label on
all of it.

The `per_block` family is the cleanest case. PTO writes **more** vector
operations than ASC on all four of `01`, `02`, `04` and `05` — and runs fewer
cycles on three of them:

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

`cast_back/05` makes the same point from the other direction: it is PTO's
**largest static win** in the set (16 operations against 8, 18 lines against 11)
and its **worst cycle regression** — **1.13x**. Shorter and slower, in one
variant.

The lesson for anyone using operation counts to choose a frontend: they measure
how much you write, which is a real and important thing, and they are not a
performance model.

---

## 9. The complete measurement

All 22 variants. `ops` and `ln` are static counts inside the vector region;
`cyc` is simulated VF cycles. Bold marks PTO using meaningfully fewer cycles
than ASC.

| variant | ops A | ops P | ln A | ln P | ln S | cyc A | cyc P | cyc S | P/A | S/A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cast_back/01_e4m3_fp32sf` | 9 | 7 | 16 | 12 | 6 | 2,105 | 2,201 | 6,908 | 1.05x | 3.28x |
| `cast_back/02_packed_ue8m0` | 12 | 12 | 12 | 16 | 9 | 2,201 | 2,233 | 7,033 | 1.01x | 3.20x |
| `cast_back/03_block_sf` | 8 | 6 | 9 | 10 | 6 | 2,080 | 2,144 | 6,969 | 1.03x | 3.35x |
| `cast_back/04_per_channel_sf` | 6 | 6 | 6 | 7 | 6 | 2,016 | 2,048 | 6,983 | 1.02x | 3.46x |
| `cast_back/05_fp4_e2m1` | 16 | 8 | 18 | 11 | 13 | 2,296 | 2,584 | 10,152 | 1.13x | 4.42x |
| `cast_back/06_col_major_compose` | 14 | 14 | 21 | 22 | 16 | 2,272 | 2,528 | 9,275 | 1.11x | 4.08x |
| `per_token/01_raw_fp32sf` | 39 | 19 | 37 | 24 | 15 | 6,176 | 4,576 | 71,606 | **0.74x** | 11.59x |
| `per_token/02_round_sf` | 45 | 14 | 43 | 20 | 17 | 4,608 | 4,688 | 71,008 | 1.02x | 15.41x |
| `per_token/03_packed_ue8m0` | 44 | 15 | 37 | 21 | 15 | 4,608 | 4,672 | 74,136 | 1.01x | 16.09x |
| `per_token/04_fp32_in_fp4_out` | 37 | 17 | 34 | 23 | 15 | 6,160 | 4,784 | 613,632 | **0.78x** | 99.62x |
| `per_token/05_col_major_sf` | 56 | 38 | 54 | 48 | 15 | 5,477 | 5,448 | 75,281 | 0.99x | 13.74x |
| `per_token/06_split_requant` | 39 | 30 | 57 | 44 | 32 | 6,864 | 4,576 | 71,606 | **0.67x** | 10.43x |
| `per_token/07_bf16_fast_compose` | 31 | 17 | 39 | 24 | 15 | 4,864 | 5,136 | 72,490 | 1.06x | 14.90x |
| `per_block/01_raw_32x32` | 21 | 22 | 21 | 25 | 18 | 1,086 | 924 | 19,276 | **0.85x** | 17.75x |
| `per_block/02_round_packed` | 27 | 30 | 28 | 33 | 20 | 918 | 936 | 19,444 | 1.02x | 21.18x |
| `per_block/03_fp4_e2m1` | 28 | 26 | 29 | 29 | 19 | 1,625 | 1,040 | 27,256 | **0.64x** | 16.77x |
| `per_block/04_col_major_tma` | 21 | 22 | 21 | 25 | 18 | 1,414 | 924 | 19,272 | **0.65x** | 13.63x |
| `per_block/05_split_compose` | 29 | 31 | 33 | 37 | 28 | 918 | 940 | 19,444 | 1.02x | 21.18x |
| `per_channel/01_raw_32tokens` | 17 | 16 | 21 | 21 | 12 | 502 | 387 | 2,677 | **0.77x** | 5.33x |
| `per_channel/02_round_packed_m` | 28 | 29 | 35 | 40 | 17 | 849 | 811 | 5,964 | **0.96x** | 7.02x |
| `per_channel/03_requant_bf16` | 23 | 19 | 29 | 35 | 14 | 930 | 721 | 4,073 | **0.78x** | 4.38x |
| `per_channel/04_compose` | 32 | 33 | 46 | 51 | 17 | 738 | 736 | 5,964 | 1.00x | 8.08x |
| **total** | **582** | **431** | **646** | **578** | **343** | **60,707** | **55,037** | **1,220,449** | | |

Aggregates:

| | PTO vs ASC | SIMT vs ASC |
|---|---|---|
| vector operations | 74% | — (no comparable count) |
| source lines | 89% | **53%** |
| simulated cycles | **0.92x mean** (0.64x–1.13x) | 14.50x mean (3.20x–99.62x) |

PTO is faster than ASC on **9 of 22** variants.

Excluding `per_token/04`, whose hand-written FP4 encoder is a special case
(section 7), SIMT's range is **3.20x–21.18x, mean 10.44x**. That is the
figure to carry away for the thread model as such; the 99.62x row measures a
missing instruction, not a programming style.

### Reading the two axes together

- **SIMT** writes **53%** of ASC's lines and costs roughly an order of
  magnitude in cycles. The win is concentrated in lane-placement work; the cost
  is lane occupancy and is paid everywhere.
- **PTO** writes **74%** of ASC's operations, **89%** of its lines, and
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
4. **The SIMT kernels are deliberately untuned.** They are written the way a GPU
   programmer writes them first, which was the comparison wanted. A separate
   SIMT-versus-SIMD suite on the same simulator, sweeping several SIMT strategies
   per kernel, found ratios from **0.33x to 53x** — that is, SIMT *winning* by 3x
   on some grouped reductions. In one case the strategy choice alone moved an
   identical kernel by a factor of two. **Nothing here should be read as "SIMT is
   slow" in general.**
5. **Static line counts exclude helpers outside the vector region**, in every
   frontend. That understates SIMT on the two FP4 variants by about 18 lines each
   — the hand-written encoder lives in a module-level helper.
6. **The group sizes here are small.** Every reduction uses `group_size = 32`,
   which is the case that leaves the most SIMT threads idle. A kernel whose
   natural parallelism matches the thread count would not look like this.

---

## 12. Conclusions

**Where SIMT genuinely simplifies the kernel.** Only where the SIMD frontends
had to move data *between lanes*. The three cases in this set are a group
broadcast (section 2), an interleave (3.1) and a transpose (3.2), and they are
also where SIMD code is at its least readable — `S.vci` plus `S.vgather2` to
express "store it transposed", or an interleave instruction plus a padded scratch
row to express "write two bytes". There SIMT is a third of the code and removes
a class of bug. Everywhere else it buys notation rather than structure, and it
pays an order of magnitude in cycles throughout.

**Where PTO's SIMD style is already clean enough.** When the operation maps onto
the hardware as written. If the data already lines up with the lanes
(`per_channel/01`: a channel *is* a lane), or the value being placed is a scalar
(`per_block/01`: a tile scale has no layout), then ASC is already saying exactly
what it means, VMI has nothing to factor out, and SIMT has no lane placement to
remove. Those variants come out 17 operations against 16, and 21 lines against
21. Reaching for a thread model there would be effort for nothing.

**What each frontend is for.**

| | choose it when |
|---|---|
| **SIMT** | getting a correct kernel quickly matters more than cycles; or the kernel is dominated by lane placement — packing, transposes, gathers — where it is both shorter and less error-prone |
| **PTO/VMI** | as the default for production vector code. On this evidence it is shorter than ASC (74% of the operations) *and* no slower (0.92x the cycles), and `size=`-parameterised helpers are reusable across widths in a way ASC structurally cannot be |
| **ASC** | an instruction VMI cannot express is needed — a per-lane variable right shift being the clearest case — or a specific distribution mode is the point |

**The methodological conclusion, which generalises past this chip.** Operation
counts and line counts measure how much you write. They were a poor predictor of
cycles here: PTO writes 74% of ASC's operations and uses 92% of its cycles, wins
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
