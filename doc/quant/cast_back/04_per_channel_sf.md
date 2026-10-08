# cast_back 04 — per-channel scales, and the broadcast disappears

New config: `sf_block` becomes $(32, 1)$. The scale now varies along $K$ and is
constant across 32 tokens:

$$
\mathrm{out}[m,k] = \mathrm{float}(q[m,k]) \cdot \mathrm{sf}\left[\left\lfloor m/32 \right\rfloor, k\right]
$$

So `sf` is $(M/32, K)$ — one scale per channel — the transpose of
[variant 01](01_e4m3_fp32sf.md)'s layout.

## Worked example

| | |
|---|---|
| `sf` | `[[1.0, 0.5, 0.25, 0.125]]` |
| `q[0, :] = 4` | `out[0] = [4.0, 2.0, 1.0, 0.5]` |
| `q[31, :] = 4` | `out[31] = [4.0, 2.0, 1.0, 0.5]` — identical |

Every token in the group uses the same per-channel scales. The scale varies along
$K$, not along $M$.

## Why this is the easy one

Every previous variant had to *construct* a scale vector, because a scale covered
32 channels while a register covers 64 lanes: two broadcast loads and a select,
every strip.

Here the scale varies per channel, and a 64-lane float32 register covers exactly
64 consecutive channels. The scale vector is already sitting in memory in
precisely the layout the register wants:

```python
ASC:  scale = S.vld(sf_ub[col])              # default NORM dist: contiguous
PTO:  scale = V.vload(sf_ub[col], size=64)   # no dist_mode, no group
```

One plain contiguous load. No `BRC_B32`, no mask, no `vsel`. The broadcast
machinery that dominated variants 01–04 is simply absent.

| | ops to build one strip's scale vector |
|---|---|
| ASC variant 01 | `vld` + `vld` + `vsel` = 3 |
| PTO variant 01 | one `brc` load with `group=2` = 1 |
| either backend, variant 04 | a plain load = 1 |

## PTO vs ASC

The two are at their closest here, for an informative reason: VMI's advantage in
the earlier variants came **entirely** from `dist_mode="brc"` + `group=` replacing
ASC's broadcast-and-select. Remove the need to broadcast and the advantage
evaporates — the plain contiguous load was never the hard case for either backend.

The generalisation, and the thing actually worth learning: **VMI helps where ASC
was forced to emulate something the hardware expresses more directly** — segmented
broadcast, segmented reduce, narrowing conversion. It does not help where ASC was
already saying exactly what it meant.

Measured: 6 operations each. A tie, and the first one in the ladder.

## GPU vs NPU — the one place the NPU wins

This is the mirror of the quantize direction. On a GPU, per-channel scales are the
*awkward* case for the matching quantize kernel, because the reduction then runs
across threads rather than within one —
`per_channel_cast_cuda.py` abandons `T.Parallel` entirely and stages partial maxima
through shared memory. Here the same granularity is the natural case, because 64
channels map one-to-one onto 64 lanes.

The general lesson: whether a layout is convenient depends on which axis maps onto
the hardware's parallel dimension, and that answer is different for
lanes-in-a-register than for threads-in-a-warp. See
[per_channel](../per_channel/README.md), where this is the whole story.

## What the harness checks

- `M` must be a multiple of 32;
- `assert_bf16_near(..., atol=0.0)` against `oracle.cast_back(..., (32, 1))`;
- the torch tier also runs `(64, 64)`, which has two token groups, so an
  implementation that ignored the group index would pass at `(32, 128)`.
