# cast_back — dequantize

Start the ladder here. `cast_back` is the only one of the four kernels with **no
reduction**: its scale factors arrive as an input, so it teaches the vector data
path before any of the reduce/broadcast machinery.

## What it computes

Given quantized values and the scales that were used to produce them, recover the
approximate original:

$$
\mathrm{out}[m,k] = \mathrm{decode}(q[m,k]) \cdot \mathrm{sf}\left[\left\lfloor m/b_m \right\rfloor, \left\lfloor k/b_k \right\rfloor\right]
$$

The **scale block** says how many tokens and how many channels share one scale:
$b_m$ tokens by $b_k$ channels. Three shapes appear in production, and the ladder
covers all of them:

| `sf_block` | scale array | one scale per | variant |
|---|---|---|---|
| $(1, 32)$ | $(M, K/32)$ | row segment of 32 channels | 01, 02, 03, 06, 07 |
| $(32, 32)$ | $(M/32, K/32)$ | 2-D tile | 04 |
| $(32, 1)$ | $(M/32, K)$ | channel, over 32 tokens | 05 |

`decode` is the identity for FP8 — the value is already a float — and a nibble
unpack for FP4. The scale itself may need decoding too, when it is stored as a
UE8M0 exponent byte rather than a float32.

## Variants

| # | config added | the idea |
|---|---|---|
| [01](01_e4m3_fp32sf.md) | FP8 in, FP32 scale, bf16 out | the VF data path end to end |
| [02](02_fp32_out.md) | float32 output | the output dtype picks the *store instruction* |
| [03](03_packed_ue8m0.md) | packed UE8M0 scales | decode an exponent byte with a shift and a mask |
| [04](04_block_sf.md) | `sf_block` $(32,32)$ | a coarser scale axis hoists the scale DMA out of the loop |
| [05](05_per_channel_sf.md) | `sf_block` $(32,1)$ | per-channel scales: the broadcast disappears |
| [06](06_fp4_e2m1.md) | FP4 (e2m1) input | 128 values per load, and widening without a convert |
| [07](07_col_major_compose.md) | column-major scales, composed | consuming a transposed layout is free |

## Running them

```bash
python -m harness.check torch/cast_back/01    # the reference, instant
python -m harness.check asc/cast_back/01      # Ascend SIMD, under the simulator
python -m harness.check pto/cast_back/01      # PTO VMI
python -m harness.check cast_back             # all 7 variants, all 3 tiers
```

## Why this kernel is the widest GPU-vs-NPU gap

`tile_kernels/quant/cast_back_cuda.py` in
[TileKernels](https://github.com/deepseek-ai/TileKernels) is 87 lines and its body
is three statements:

```python
for id in T.Parallel(sf_size_aligned):
    sf_shared[i, j] = transform_sf(load_sf(x_sf, ...), in_config)
for i, j in T.Parallel(TILE_M, TILE_K):
    out_fragment[i, j] = x_shared[i, j] * sf_shared[i // num_per_tokens,
                                                    j // num_per_channels]
```

`cast_back_asc.py` is 203 lines for the same contract. Because the algorithm is
so nearly trivial, every line of that difference is hardware, which makes this the
best place to see what the difference consists of:

- the scale index `sf_shared[i // 32, j // 32]` becomes a broadcast **load mode**,
  plus a select wherever the scale group is narrower than the register (01);
- `transform_sf` — a six-line scalar macro on the GPU — becomes an in-register
  shift-and-mask on a whole vector, with three different strategies depending on
  the scale layout (03, and production's lazy / `e2b` paths);
- the FP4 unpack and the bfloat16 pack, both implied by buffer dtypes on the GPU,
  become distribution modes and a `vintlv` idiom (06);
- the loop nest has to be restructured when the scale granularity changes, because
  the DMA placement depends on it (04).

And one place where they agree exactly: variant 05. Per-channel scales are a plain
contiguous load on the NPU and a plain index expression on the GPU, because nothing
has to move between lanes or threads.

## Further reading

- [`doc/quant/README.md`](../README.md) — the shared quantization maths
- [`doc/vf-lane-widths-and-limits.md`](../../vf-lane-widths-and-limits.md) — the
  256-byte register, and why 32 is not a legal vector width
- [`doc/pto-vs-asc.md`](../../pto-vs-asc.md) — the ASC/VMI comparison with measurements
