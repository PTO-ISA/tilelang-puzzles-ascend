# cast_back 04 — one scale per 32x32 tile

New config: `sf_block` becomes $(32, 32)$. The scale index gains a row term:

$$
\mathrm{out}[m,k] = \mathrm{float}(q[m,k]) \cdot \mathrm{sf}\left[\left\lfloor m/32 \right\rfloor, \left\lfloor k/32 \right\rfloor\right]
$$

Numerically that is the whole change. But it changes the **schedule**, and that is
the point of this variant.

## Worked example

A `(32, 64)` tensor is two 32x32 tiles, so two scales:

| | |
|---|---|
| `sf` | `[[0.5, 0.25]]` |
| `q[0, 0] = 4` | tile (0,0), scale 0.5 → `out = 2.0` |
| `q[0, 32] = 4` | tile (0,1), scale 0.25 → `out = 1.0` |
| `q[31, 0] = 4` | tile (0,0) again, 31 rows away → `out = 2.0` |

All 32 rows of a tile share one scale. Compare
[variant 05](05_per_channel_sf.md), where the sharing runs the other way.

## A coarser scale axis means less DMA

With per-token scales every token needed its own scale row, so the scale copy sat
inside the token loop. Now 32 consecutive tokens share one row, so the copy hoists
out:

```python
for token_block in ...:
    T.copy(Sf[token_block, 0], sf_ub)      # once per 32 tokens
    for row in T.serial(32):
        ...                                 # reuse sf_ub
```

That is 32x fewer scale DMAs. Production expresses the same insight differently —
it gives the scale buffer a smaller multi-buffer depth than the value buffer
(`sf_stages = num_stages if num_per_tokens == 1 else 1` in `cast_back_asc.py`):
when the scale changes once per tile there is no point double-buffering it.

Restructuring the loop nest to match the scale granularity is the kind of change
that does not exist at all in the torch tier, where the scale is just an index
expression. It is most of what writing these kernels consists of.

## PTO vs ASC — no difference at all

Worth stating explicitly, because it locates where the ASC/VMI choice actually
applies. This variant changes the loop nest and the DMA placement, both of which
live **outside** `T.SimdVF()` — in the `@T.prim_func` schedule, which is identical
in the two backends. The vector body is [variant 01](01_e4m3_fp32sf.md)'s,
unchanged.

The ASC/VMI choice is a choice of *vector instruction set*, not of programming
model. Everything about tiling, buffering, DMA and core assignment is shared.
Production's PTO port touched only the `T.SimdVF` bodies for exactly this reason —
every schedule and host contract came through byte-identical.

Measured: 8 operations for ASC, 6 for PTO — and the 2 saved are the variant-01
broadcast, not anything this variant introduced.

## GPU vs NPU

On a GPU the scale granularity is an index expression and the compiler handles the
rest; there is no DMA to hoist because there is no explicit staging buffer. Here
the granularity decides the loop nest.

## What the harness checks

- a divisibility guard: `M` and `K` must both be multiples of 32, asserted with a
  message rather than failing deep inside the kernel;
- `assert_bf16_near(..., atol=0.0)` against
  `oracle.cast_back(..., (32, 32))` — byte-exact;
- the torch tier additionally runs `(64, 64)`, which has two tile *rows*, so an
  implementation that ignored the row term would pass at `(32, 128)` and fail here.
