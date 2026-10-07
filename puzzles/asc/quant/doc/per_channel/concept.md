# per_channel — one scale per channel, across 32 tokens — ASC tier

## The maths

```
    amax = max(|x|) over 32 tokens, for each channel independently
    sf   = max(amax, 1e-4) / 448                 (optionally rounded up to 2^k)
    q    = cast_e4m3(x * (448 / amax))

    x: (M, K)   q: (M, K)   sf: (M/32, K)

The reduction runs along **M**. That one change inverts the difficulty between
backends.
```

Implemented in plain PyTorch in `puzzles/torch/quant/answer/per_channel/`, which is
what every kernel here is checked against.

## The variant ladder

| # | config | new idea |
|---|---|---|
| 01 | raw float32 scale | reducing along M: no cross-lane reduce, no broadcast |
| 02 | `round_sf` + UE8M0 packed along **M** | the only place packing costs an instruction |
| 03 | requant (`with_sf`) | both broadcast patterns in one kernel; exact ties |
| 04 | bfloat16 reduction, composed | the max can stay on the integer unit |

Each variant's own docstring is its implementation guide — the instruction-level
reasoning lives next to the code it describes, where it cannot drift away from it.
Start with `01`.

## GPU vs NPU

**This is the kernel where the NPU wins, and it is the most useful comparison in
the repo.**

On a GPU, consecutive threads hold consecutive *channels* of one token. Reducing
along M means combining values held by different threads, so
`per_channel_cast_cuda.py` abandons `T.Parallel` entirely — the only kernel in the
quant family that does. It stages partial maxima in shared memory, calls
`T.sync_threads()`, then has one owner thread per channel do the combining:

```python
for i in T.unroll(VEC_K):
    amax_shared[i, tid] = amax_local[i]
...
owner_tid = sf_col % num_threads_per_token
for i in T.serial(num_row_slices_per_sf_row):
    amax_var = T.max(amax_var, amax_shared[owner_offset, src_tid])
T.sync_threads()
```

with comments about bank conflicts and store coalescing.

On this NPU, 64 channels sit in 64 float32 lanes, one per lane. Reducing 32 tokens
is 32 element-wise `vmax` operations between whole registers — no masks, no
`vcmax`, no cross-lane anything. And the resulting scale vector is already laid out
one value per lane, so applying it is a plain contiguous load: the broadcast
machinery that dominates `per_token` never appears.

The generalisation worth taking away: **whether a reduction axis is cheap depends
on which axis maps onto the hardware's parallel dimension.** Lanes-in-a-register
and threads-in-a-warp give opposite answers. Deciding a tensor layout without
knowing which you have is how kernels end up slow for structural reasons.

One NPU-specific cost does appear, in variant 02: the scale bytes pack along M,
which is *not* the fastest-varying axis, so the packing needs a real `vintlv`
instead of being a host-side `.view()`. On a GPU the pack factor is 4 rather than 2
and the same asymmetry exists, but it is handled by the index expression.

## Further reading

- `doc/vf-lane-widths-and-limits.md` — the register geometry and legal widths
- `doc/known-issues.md` — toolchain limits, each with a reproducing script
- `doc/gpu-vs-npu.md` — the consolidated GPU comparison
