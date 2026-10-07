# cast_back — dequantize — ASC tier

## The maths

```
    out[m, k] = decode(q[m, k]) * sf[block(m, k)]

No reduction: the scale factors are an **input**. That is why this kernel comes
first in the ladder — it teaches the vector data path with none of the
reduce/broadcast machinery.
```

Implemented in plain PyTorch in `puzzles/torch/quant/answer/cast_back/`, which is
what every kernel here is checked against.

## The variant ladder

| # | config | new idea |
|---|---|---|
| 01 | `sf_block=(1,32)`, FP8 in, bf16 out | the VF data path end to end |
| 02 | float32 output | the output dtype changes the *store instruction* |
| 03 | packed UE8M0 scales | decode an exponent byte with a shift and a mask |
| 04 | `sf_block=(32,32)` | a coarser scale axis hoists the scale DMA out of the loop |
| 05 | `sf_block=(32,1)` | per-channel scales: the broadcast disappears entirely |
| 06 | FP4 (e2m1) input | 128 values per load, and widening without a convert |
| 07 | column-major scales, all composed | consuming a transposed layout is free |

Each variant's own docstring is its implementation guide — the instruction-level
reasoning lives next to the code it describes, where it cannot drift away from it.
Start with `01`.

## GPU vs NPU

`cast_back_cuda.py` in TileKernels is 87 lines and its body is three statements:

```python
for id in T.Parallel(sf_size_aligned):
    sf_shared[i, j] = transform_sf(load_sf(x_sf, ...), in_config)
for i, j in T.Parallel(TILE_M, TILE_K):
    out_fragment[i, j] = x_shared[i, j] * sf_shared[i // num_per_tokens,
                                                    j // num_per_channels]
```

`cast_back_asc.py` is 203 lines for the same contract. Everything in the
difference is hardware, and this kernel is the best place in the repo to see what
that consists of, because the algorithm is so nearly trivial:

- the scale index `sf_shared[i // 32, j // 32]` becomes a broadcast *load mode*
  plus, where the group is narrower than the register, a select (variant 01);
- `transform_sf` — a six-line scalar macro on the GPU — becomes an in-register
  shift-and-mask on a vector, and three different strategies depending on the
  scale layout (variant 03, and production's `use_e2b_scale` / lazy paths);
- the FP4 unpack and the bfloat16 pack, both implied by buffer dtypes on the GPU,
  become distribution modes and a `vintlv` idiom (variant 06);
- the loop nest has to be restructured when the scale granularity changes, because
  the DMA placement depends on it (variant 04).

What does *not* differ: variant 05. Per-channel scales are a plain contiguous load
on the NPU and a plain index expression on the GPU, because nothing has to move
between lanes or threads.

## Further reading

- `doc/vf-lane-widths-and-limits.md` — the register geometry and legal widths
- `doc/known-issues.md` — toolchain limits, each with a reproducing script
- `doc/gpu-vs-npu.md` — the consolidated GPU comparison
