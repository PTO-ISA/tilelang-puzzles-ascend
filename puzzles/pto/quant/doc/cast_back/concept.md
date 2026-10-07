# cast_back — dequantize — PTO tier

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

`cast_back_cuda.py` in [TileKernels](https://github.com/deepseek-ai/TileKernels) is 87 lines and its body is three statements:

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

## PTO vs ASC

The clearest win in this kernel is **variant 06**, FP4 input.

An unpacking load yields 128 values; the scale is float32 at 64 lanes per register.
ASC must split the strip into two 64-lane halves and duplicate the whole
multiply-convert-store path, because its lane count lives in the dtype string it
reinterprets through — `"float32x64"` cannot name the 128-lane case. VMI has a
128-lane logical float32, so the split never happens: 16 vector operations become
8.

**Variant 01** shows the other mechanism: a 64-lane strip spans two 32-channel
groups, so ASC broadcasts each scale separately and stitches with a predicate
(3 operations) where VMI states the intent (1):

```python
# ASC
lo = S.vld(sf_ub[g], dist="BRC_B32"); hi = S.vld(sf_ub[g+1], dist="BRC_B32")
scale = S.vsel(lo, hi, mask_low)
# PTO
scale = V.vload(sf_ub[g], size=64, stride=1, dist_mode="brc", group=2)
```

**Variants 03, 05 and 07 are draws**, and the files say so rather than inventing
an advantage. 03 is the same bit trick on both sides — VMI only drops the lane
count from the reinterpret — and it is also where VMI is the *weaker* surface:
it has no per-lane variable shift, so production's PTO path computes both byte
positions and selects where ASC applies a shift vector. 05 is a plain contiguous
load in both. 07 is 14 operations each: VMI's grouped scale load saves one, and
ASC amortises its shift setup across both halves, and the two cancel.

See `doc/pto-vs-asc.md` for the consolidated comparison and
`python tools/vf_lines.py` for the per-variant measurements.

## Further reading

- `doc/vf-lane-widths-and-limits.md` — the register geometry and legal widths
- `doc/known-issues.md` — toolchain limits, each with a reproducing script
- `doc/gpu-vs-npu.md` — the consolidated GPU comparison
