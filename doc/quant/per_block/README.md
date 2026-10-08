# per_block — one scale per 32x32 tile

The weight quantization granularity. Do [per_token](../per_token/README.md) first:
the arithmetic is the same and this kernel's novelty is that the reduction is
**two-dimensional**.

## What it computes

$$
\mathrm{amax}[i,j] = \max_{(m,k) \in T_{ij}} \left\lvert x[m,k] \right\rvert
$$

where $T_{ij}$ is the 32x32 tile at row-block $i$, column-block $j$. Then the
scale and the quantized values exactly as in `per_token`:

```
x  : (M, K)        q : (M, K)        sf : (M/32, K/32)
```

1024 values per scale instead of 32, so the scale array is 1024x smaller than the
data. Weights are reused across every token in a batch, so their scales are
amortised and can afford to be coarse.

## The accuracy cost is smaller than it looks

A 32x32 tile spans 32 tokens, so one outlier anywhere in it inflates the scale for
all 1024 values. The usual conclusion — that this costs a lot of accuracy — does
**not** follow for FP8, and the measured numbers say so:

Round-trip error, mean over 8 seeds at `(64, 256)`:

| | e4m3 | e2m1 |
|---|---|---|
| per_token (32 values/scale) | 2.9% | 12.6% |
| per_block (1024 values/scale) | 3.5% | 15.0% |

e4m3 is a *floating-point* target, so a larger scale mostly shifts the exponent and
relative precision survives. The real cost is **underflow**: e4m3's smallest
subnormal is `2^-9`, so once the scale is large enough the tile's smallest values
quantize to zero and are gone. With a 4000.0 outlier planted in a tile, `per_block`
loses 5 of the other 1023 values to underflow and `per_token` loses none.

(For an *integer* target like int8, precision is absolute and the outlier argument
is much stronger. That is the context it usually gets quoted in, and it does not
transfer unchanged to FP8.)

FP4 widens the granularity gap about fourfold — 0.6 points of error between the
granularities at e4m3 against 2.4 points at e2m1 — which is why production pairs
FP4 with fine granularity for activations and reserves coarse blocking for weights.
(Single-seed measurements of these numbers scatter by half a point or so, which is
why the table is a mean over 8 seeds and `harness/doc_examples.py` re-measures it.)

## Why the tile's own width is unusable

A 32x32 tile is 32 values wide, and 32 looks like the natural vector width. It is
not available: the legal lane counts are `{1, 2, 4, 8, 64, 128, 256}` and 32 is
absent — half a 256-byte register, neither one 32-byte slice nor one whole
register.

The next guess, 8 lanes (one 32-byte slice), **does not compile** for the
bfloat16-to-float32 convert this kernel needs:

```
VMI-LAYOUT-CONTRACT: pto.vmi.extf has no registered layout support
```

`harness/probe/vf_lane_limits.py` reproduces it. So this kernel — and production's —
flattens the tile into one 1024-value run and reduces 64 or 128 lanes at a time.

**The transferable rule: pick the lane width from the hardware, then reshape the
problem to fit it, not the other way round.** A GPU never asks this question.

## Variants

| # | config added | the idea |
|---|---|---|
| [01](01_raw_32x32.md) | raw float32 scale | the 2-D reduction, and the lane-width rule |
| [02](02_round_packed.md) | `round_sf` + packed UE8M0 | the same exponent trick, one scale per tile |
| [03](03_fp4_e2m1.md) | FP4 output | the coarsest granularity meets the coarsest format |
| [04](04_col_major_tma.md) | column-major scales | **free here**, unlike per_token/05 |
| [05](05_split_compose.md) | `sf_only` / `cast_only`, composed | `cast_only` is bit-exact with a pow2 scale |

## Where PTO buys the least

`python tools/vf_lines.py` puts PTO's *static* operation count slightly **higher**
than ASC's on four of five variants here. Two reasons, both real: the reduction is
a whole-vector `group=1` reduce, so the segmented-operation advantage that drives
`per_token` does not apply; and VMI requires explicit `size=` and mask operands.

The static count also *understates* PTO, in the opposite direction: PTO reduces 128
lanes per iteration against ASC's 64, so it runs **8 iterations rather than 16**
and issues fewer instructions at runtime. A count of operations *written* cannot
see a count of operations *executed*.

The honest summary for this kernel: VMI is not shorter here, it is wider.
