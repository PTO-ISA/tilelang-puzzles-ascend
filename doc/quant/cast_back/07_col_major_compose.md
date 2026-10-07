# cast_back 07 — column-major scales, everything composed

Final `cast_back` variant. Composed config:

```
packed UE8M0 scales + column-major layout + FP4 values -> bfloat16 output
```

## Column-major scales turn out to be free here

The scale array is stored transposed — `sf_cm[word, token]` rather than
`sf_packed[token, word]`. The *public* shape stays $(M, K/32)$; the host just
takes a `.T` view.

The reason is the consumer: the GEMM processes a tile of tokens at a time and
wants that tile's scales contiguously, so a bulk async copy can fetch them. In
row-major order one group's scales across 32 tokens are strided by $K/32$; in
column-major they are adjacent.

For *this* kernel that costs nothing, and the reason is worth noticing. The scale
load is a **broadcast** — it reads one scalar and fans it across the register —
and a broadcast does not care about the stride between consecutive elements,
because it only ever touches one. So the transpose is absorbed entirely by
swapping two index expressions:

```python
row-major     S.vld(sf_ub[token, word], dist="BRC_B16")
column-major  S.vld(sf_ub[word, token], dist="BRC_B16")
```

The expensive direction is the other one: *producing* column-major scales in a
quantize kernel means transposing values that are spread across lanes, which needs
an in-register gather. That is
[per_token/05](../per_token/05_col_major_sf.md), the hardest variant in the ladder.
**Consuming them is trivial; producing them is not.**

## Footprint

At `M=32, K=128`, composing FP4 values with packed UE8M0 scales:

| | bytes |
|---|---|
| values as bfloat16 | 8192 |
| values as packed FP4 | 2048 |
| scales as float32 | 512 |
| scales as packed UE8M0 | 128 |
| **total, bf16 + fp32 scales** | **8704** |
| **total, FP4 + packed scales** | **2176** |

4x smaller, which is the point of having both features in the ladder.

## Scheduling: the scale tile

The scales for one token are now a *column*, so a whole tile of them is DMA'd in
one go and indexed per token, rather than copied per token. Tokens are processed in
blocks of 32 so the UB scale buffer has a static shape.

## PTO vs ASC — where the measurement says "no difference"

**This variant is a draw: 14 operations each.** That is worth understanding rather
than glossing.

An FP4 unpacking load gives 128 values. In ASC the scale machinery works a register
at a time, so the strip is processed as two 64-lane halves and the whole
scale-decode sequence runs **twice** per strip. VMI keeps 128 lanes as one logical
vector, so it runs once — using two features, both the same "width as an argument"
idea:

1. **`dist_mode="brc"` with `group=2` and a stride** fetches *two* packed words,
   broadcasting each across 64 lanes, in one load. ASC's `BRC_B16` broadcasts a
   single scalar, so it needs one load per word.
2. **`V.create_mask(32, size=128, group=2)`** builds the repeating "first 32 of
   every 64 lanes" predicate that selects shift 23 vs shift 15 across all 128
   lanes at once. ASC's `S.pset(32, "PAT_VL32")` is a 64-lane pattern from a fixed
   catalogue; there is no 128-lane form of it.

So why is it a draw? Because ASC hoists its shift-pattern setup out of the strip
loop and then reuses it for *both* halves, so the per-half cost it pays is small.
PTO's single-pass form avoids the duplication and pays for a wider setup instead.
The two roughly cancel.

Where the width advantage really shows is [variant 06](06_fp4_e2m1.md), which
saves 8 of 16 operations. The difference between the two cases: in 06 it is the
*value* path that ASC has to split, with the whole multiply-convert-store sequence
duplicated, whereas here the duplicated region is a two-instruction scale decode.
**VMI's advantage pays in proportion to how much work sits inside the region ASC
has to duplicate.**

### A measurement note

The operation count excludes bit reinterpretation, which emits no instruction. ASC
spells it `T.reinterpret` and VMI spells it `V.vinterpret_cast`, so counting the
VMI form alone would penalise PTO for a purely notational difference — it put this
variant at +2 before `tools/vf_lines.py` was corrected.

## Where this lands relative to production

`cast_back_asc.py` adds, beyond this file: multiple vector cores with a persistent
loop, double-buffered UB, a UB-aliasing trick where the packed and decoded scale
buffers are the same allocation, and two further scale-decode strategies
(`use_e2b_scale` for a one-load-fans-256-lanes fast path, and a separate route for
per-channel scales packed along M). None of them changes the arithmetic here.

## What the harness checks

- `M` must be a multiple of 32 (the scale tile is 32 tokens wide);
- `assert_bf16_near(..., atol=0.0)` against
  `oracle.cast_back(..., packed=True, fp4=True)` — byte-exact with both
  memory-saving features composed;
- the e2m1 code table, via `doc_examples`.
