# per_channel 01 — reduce across 32 tokens, one scale per channel

See the [kernel overview](README.md) for why this axis is awkward. This page is
about what the reduction becomes when the axis is the slow one.

## Worked example

A `(32, 64)` tensor, one token group, 64 channels. Put the two largest values in
different rows of different columns:

| | |
|---|---|
| `x[0, 0] = 4.0` | channel 0's maximum, in row 0 |
| `x[17, 1] = 2.0` | channel 1's maximum, 17 rows away |
| `sf[0, 0]` | `4.0/448` |
| `sf[0, 1]` | `2.0/448` — a *different* scale, same row |
| `q[0, 0]` | `448` |
| `q[17, 1]` | `448` |

Both maxima land on 448 because each channel got its own scale. Contrast
[per_token/01](../per_token/01_raw_fp32sf.md), where everything in a row shared one
scale.

## The reduction is an elementwise max

One vector load brings 64 **channels** of one token. Those 64 values belong to 64
different output scales, so they must stay in their lanes — there is nothing to
reduce *within* the vector. The reduction is across loads instead:

```python
acc = S.vabs(S.vld(x_ub[0, col]))
for row in range(1, BLOCK_MN):
    acc = S.vmax(acc, S.vabs(S.vld(x_ub[row, col])))
```

32 loads, 32 `vmax`, and lane `c` independently tracks channel `c`'s maximum. No
`vcmax`, no masks, no segment arithmetic — **simpler than `per_token`, because the
layout happens to match what the lanes want.**

That is the general shape of it: a reduction along the vectorized axis needs
in-register reduction machinery (masks, segments, lane groups); a reduction across
the vectorized axis needs none, because the loop does the reducing.

## The broadcast disappears too

`per_token` had to broadcast one scalar scale to 64 lanes. Here the 64 scales are
*already* a vector in the right lanes, and every one of the 32 rows uses that same
vector:

```python
for row in range(BLOCK_MN):
    q = S.vcvt(S.vmul(S.vld(x_ub[row, col]), inv), T.float8_e4m3fn)
```

So the reciprocal is computed once and reused 32 times. The `BRC_B32` loads and
`vsel` masks that dominate `per_token`'s apply pass are simply absent.

## PTO vs ASC — a genuine draw, and it is worth saying why

Both backends write the same loop. VMI's two structural advantages do not apply:

- **`group=` buys nothing.** There is no segment structure to express — the
  reduction is whole-lane-wise, which both backends already do in one operation.
- **Width parameterisation buys little**, because the kernel needs exactly one
  width.

```python
ASC: acc = S.vmax(acc, S.vabs(S.vld(x_ub[row, col])))
PTO: acc = V.vmax(acc, V.vabs(V.vload(x_ub[row, col], size=64), mask), mask)
```

Measured: 17 ASC operations against 16, and **21 source lines each** -- the closest
thing to a tie in the whole ladder. VMI's explicit masks cost it the characters that
its slightly tighter operation set wins back.

**This is the clearest "no advantage" case in the repo, and it is informative
rather than disappointing.** VMI's wins come from expressing *segmented* behaviour
and from width being an argument. A kernel whose reduction is already one
instruction per vector has neither to offer. Knowing that in advance is what tells
you a port is not worth doing.

One thing PTO *could* exploit and this variant does not: a 128-lane load would
halve the loop's instruction count by covering 128 channels per step. That is
[variant 04](04_compose.md).

## GPU vs NPU

`T.reduce_max(y_abs, y_amax, dim=0)` instead of `dim=1`, plus a different
`T.Fragment` layout annotation so the reduction stays within a warp. The kernel
body is otherwise unchanged — the compiler re-assigns elements to threads.

Here the axis chooses the *instruction*, so the two kernels have genuinely
different shapes. The overview's table lays out the four consequences.

## What the harness checks

- `M % 32 == 0` and `K % 64 == 0`, with messages naming the reason;
- scales within 1 ULP on the NPU tiers, exact on the torch tier;
- FP8 values via `assert_fp8_near`;
- the torch tier sweeps `(32,128)`, `(64,64)` and `(32,256)`. `(64,64)` has **two**
  token groups, so an implementation that reduced over all rows instead of
  per-group would pass at `(32,128)` and fail there;
- `harness/doc_examples.py` re-measures the overview's granularity table on this
  variant, so the claim that the axis costs no accuracy cannot rot.
