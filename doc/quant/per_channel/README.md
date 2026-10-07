# per_channel — one scale per channel, reduced across tokens

The third granularity. Do [per_token](../per_token/README.md) first; the arithmetic
is unchanged and the novelty is the **axis**.

## What it computes

The reduction runs down a column — across tokens — for each channel:

$$
\mathrm{amax}[g,c] = \max_{m \in G_g} \left\lvert x[m,c] \right\rvert
$$

where $G_g$ is a group of 32 consecutive tokens. Then `sf = amax/448` and
`q = x * 448/amax` as always.

```
x  : (M, K)        q : (M, K)        sf : (M/32, K)
```

So the scale array has **one entry per channel**, not per group of channels. For
the shapes used here that makes it $K/32$ times *larger* than `per_token`'s — this
granularity is finer in the channel direction and coarser in the token direction.

## Why this axis is the awkward one

Everything about the hardware favours the fastest-varying axis, and this kernel
reduces across the slow one.

| | `per_token` | `per_channel` |
|---|---|---|
| reduce along | $K$ — contiguous | $M$ — stride $K$ |
| one vector holds | 64 values of one group | 64 **channels** of one token |
| reduction becomes | an in-register `vcmax` | an **elementwise** `vmax` across 32 loads |
| scale broadcast | one scalar to 64 lanes | a *vector* reused by 32 rows |

The in-register reduction disappears entirely. `vcmax` reduces *within* a vector,
but here the 64 values in a vector belong to 64 different channels and must stay
separate. So the kernel accumulates with plain elementwise `vmax` over 32 row
loads, and each lane independently tracks its own channel's maximum.

That is simpler to write than `per_token`'s masked segment reduction. The cost
moves elsewhere: the scales now pack along $M$, a direction that is **not adjacent
in memory**, which is what [variant 02](02_round_packed_m.md) is about.

## Where it is used

Weight-side quantization for the transposed GEMM, and the requantization step
between two layers whose granularities differ — which is
[variant 03](03_requant_bf16.md), and the only variant in the repo whose FP8 output
is deliberately not bit-exact.

## Variants

| # | config added | the idea |
|---|---|---|
| [01](01_raw_32tokens.md) | raw float32 scale | reduction along $M$ as an elementwise max |
| [02](02_round_packed_m.md) | `round_sf` + packing **along M** | the one place packing needs a real instruction |
| [03](03_requant_bf16.md) | requantize from per-token input | exact ties, proven rather than tolerated |
| [04](04_compose.md) | the composed bfloat16 path | 128 channels per step |

## The accuracy picture

Reducing over 32 tokens with one scale per channel sits between the other two
granularities, and the measured round-trip errors say so:

Mean over 8 seeds at `(64, 256)`:

| granularity | values per scale | e4m3 round-trip |
|---|---:|---|
| per_token | 32 | 2.9% |
| per_channel | 32 | 2.9% |
| per_block | 1024 | 3.5% |

`per_token` and `per_channel` share the same *count* of values per scale and come
out indistinguishable — **the axis matters for performance, not for accuracy.**
What costs accuracy is how many values share a scale, which is `per_block`'s story.

For independent inputs this is what one should expect: a channel's 32 values across
tokens and a token's 32 values across channels are the same distribution, so their
maxima are too. Worth stating because single-seed measurements of these three
numbers scatter by several tenths of a point and can easily suggest an ordering
that is not there -- which is why this table is a mean and
`harness/doc_examples.py` re-measures it.

## GPU vs NPU

`T.reduce_max(..., dim=0)` versus `dim=1` is a one-character change on a GPU: the
fragment layout annotation absorbs it, and the compiler re-assigns elements to
threads. There is no equivalent here — the axis determines which *instruction*
does the reduction (`vcmax` within a register against `vmax` across loads), so the
two kernels have different shapes rather than different arguments.
