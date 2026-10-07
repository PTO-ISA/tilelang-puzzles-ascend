# per_token — one scale per 32 channels of a token — ASC tier

## The maths

```
    amax = max(|x|) over each group of 32 channels
    sf   = max(amax, 1e-4) / 448                 (optionally rounded up to 2^k)
    q    = cast_e4m3(x * (448 / amax))

    x: (M, K)   q: (M, K)   sf: (M, K/32)

The activation quantization granularity: fine enough that one outlier token cannot
inflate another's scale.
```

Implemented in plain PyTorch in `puzzles/torch/quant/answer/per_token/`, which is
what every kernel here is checked against.

## The variant ladder

| # | config | new idea |
|---|---|---|
| 01 | raw float32 scale | **the reduction**: three passes, two memory barriers |
| 02 | `round_sf` | the ceil-log2 exponent trick, integer-only |
| 03 | `use_packed_ue8m0` | the scale as one byte; the int16 pairing is free here |
| 04 | float32 in, FP4 out | no float32→e2m1 instruction; round-to-odd intermediate |
| 05 | column-major scales | an in-register transpose: `vci` + `vgather` |
| 06 | `sf_only` / `cast_only`, requant | trace-time modes; why `cast_only` can differ |
| 07 | bfloat16 compute, composed | production's hot loop |

Each variant's own docstring is its implementation guide — the instruction-level
reasoning lives next to the code it describes, where it cannot drift away from it.
Start with `01`.

## GPU vs NPU

On a GPU the reduction is a library call:

```python
T.copy(y_ub, y_local)
for i, j in T.Parallel(block_groups, group_size):
    y_abs_local[i, j] = T.abs(y_local[i, j])
T.reduce_max(y_abs_local, y_amax_local, dim=1)
```

The compiler decides whether that is intra-thread, intra-warp or through shared
memory, and inserts the synchronisation. On the NPU you choose the reduction's lane
extent with a predicate, place its result in Unified Buffer yourself, and write the
barriers — `S.mem_bar("VST_VLD")`, twice, and omitting either silently reads stale
data.

Three further GPU/NPU contrasts are sharpest in this kernel:

- **Layout annotations vs lane arithmetic.** The CUDA version tunes
  `T.annotate_layout(... forward_thread_fn ...)` so that loads coalesce. The NPU
  version has no such knob: the mapping from memory to lanes *is* the distribution
  mode you wrote.
- **The transpose (variant 05).** `out_sf[g, m] = ...` on a GPU is an index change;
  any thread can write anywhere. A register lane cannot move, so the NPU computes
  per-lane source indices from `S.vci` and gathers.
- **bfloat16 compute (variant 07).** On a GPU, computing in a narrower type is a
  dtype choice. Here it doubles the lanes per register and therefore changes the
  loop structure, the reduction strategy and the abs implementation (a bitwise AND
  on the integer unit).

## Further reading

- `doc/vf-lane-widths-and-limits.md` — the register geometry and legal widths
- `doc/known-issues.md` — toolchain limits, each with a reproducing script
- `doc/gpu-vs-npu.md` — the consolidated GPU comparison
