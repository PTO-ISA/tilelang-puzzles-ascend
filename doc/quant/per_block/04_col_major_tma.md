# per_block 04 — column-major tile scales

The same config as [per_token/05](../per_token/05_col_major_sf.md), and the reason
to do it here is that **it is nearly free on this kernel while it was the hardest
variant in `per_token`**. That contrast is the lesson.

## Why it is free here

`per_token` had to transpose a *vector* of scales: the kernel computed 64 scales in
one register and had to scatter them down a column. A register's lanes cannot move,
so it needed `S.vci` to build an index vector and a `vgather` to permute — a whole
page of lane arithmetic.

`per_block` produces **one scalar per tile**. A single value has no layout, so
writing it to `SfCm[kb, mb]` instead of `Sf[mb, kb]` is a change of two indices:

```python
Sf:   T.Tensor((num_m_blocks, num_k_blocks), T.float32)   ->  Sf[mb, kb]
SfCm: T.Tensor((num_k_blocks, num_m_blocks), T.float32)   ->  SfCm[kb, mb]
```

No transpose instruction, no gather, no index vector. The kernel is otherwise
identical to [variant 01](01_raw_32x32.md).

**The transferable point: the cost of a layout change depends on how much data one
vector operation produces, not on the layout itself.** Coarse granularity makes
layout changes cheap for exactly the same reason it makes scales cheap.

## Why the consumer wants it

A GEMM reading these scales wants one tile-column's scales contiguous, so it can
fetch them with a single bulk async copy while the matrix tiles stream in. The
`_tma` in the variant name is the GPU term for that bulk copy; the NPU equivalent
is the DMA that `T.copy` emits.

[cast_back/07](../cast_back/07_col_major_compose.md) shows the consuming side, where
a broadcast load does not care about stride at all.

## PTO vs ASC

Nothing specific to this variant — the two differ exactly as they do in
[variant 01](01_raw_32x32.md) (128-lane reduction against 64), because the config
only changes two index expressions and both backends express that identically.

Measured: 21 ASC operations against 22, the same as variant 01.

## GPU vs NPU

This is the config where the two platforms agree, and it is worth saying why. On a
GPU `out_sf[kb, mb] = s` is a different address for a value that is already in a
register — free. On the NPU it is also free, because the value is a scalar in lane
0 of a register and the store targets one address.

The platforms diverge only when the *layout of a vector* has to change — which is
`per_token/05`, free on the GPU and expensive here. Comparing the two variants is
the cleanest demonstration in the repo of where the thread model actually buys
something.

## What the harness checks

- the shape really is transposed, with an explicit message:
  ```python
  assert sf_cm.shape == (k // BLOCK_K, m // BLOCK_MN)
  ```
- `sf_cm.T` matches the row-major oracle within 1 ULP;
- FP8 values via `assert_fp8_near`;
- the torch tier runs `(64, 128)` and `(128, 256)` — both non-square in tile counts,
  so an implementation that swapped the two dimensions consistently would still be
  caught.
