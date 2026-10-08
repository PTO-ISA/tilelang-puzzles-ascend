# GPU vs NPU: the same four kernels, two programming models

Every NPU kernel file in this repo carries its own **GPU vs NPU** section with the
specifics for that variant. This document is the overview: what the general
difference is, and where it does and does not bite.

Reference points, both public:
[TileKernels](https://github.com/deepseek-ai/TileKernels) has `*_cuda.py` and
`*_asc.py` implementations of all four kernels side by side; the GPU puzzle ladder
these exercises were modelled on is tilelang's own `examples/` plus an internal
`tk_quant` ladder with the same 01→09 feature ordering.

## The one difference everything follows from

```python
# GPU: name one element of an index space. The compiler assigns it to a thread
#      and decides the load width.
for i, j in T.Parallel(block_m, block_k):
    out[i, j] = x[i, j] * sf[i, j // 128]

# NPU: name one vector register. You decide the width, the layout, the staging.
with T.SimdVF():
    for strip in T.serial(hidden // 64):
        x = S.vcvt(S.vld(x_ub[strip * 64], dist="UNPK_B16"), T.float32)
        ...
```

`T.Parallel` is a *parallel-for over elements*. `T.SimdVF` is a *vector-register
scope*. The first is closer to writing CUDA; the second is closer to writing
AVX-512 intrinsics (see `vf-lane-widths-and-limits.md` for that analogy, which is
worth taking seriously).

Concretely, `cast_back_cuda.py` is 87 lines for the same contract that takes
203 lines of `cast_back_asc.py` — and the CUDA body is three statements.

## What the NPU makes explicit that the GPU hides

| concern | GPU | NPU |
|---|---|---|
| unit of work | one element `(i, j)` | one vector of 64 / 128 / 256 lanes |
| vectorisation | compiler picks the width | you pick it; illegal widths are a compile error |
| staging | `alloc_fragment` (registers), `alloc_shared` (SMEM) | `alloc_shared` is **Unified Buffer**; registers are reached by `vld` |
| multi-buffering | compiler / pipeliner | `T.annotate_buffer_versions({buf: 2})` by hand |
| reduction | `T.reduce_absmax(frag, out, dim=1)` | masked `vcmax`, or `vcmax(..., group=n)`, then stage partials through UB |
| broadcast | `sf[i, j // 32]` as an index expression | a broadcast *load mode*, plus a select if the group is narrower than the register |
| dtype conversion | assignment between typed buffers | `vcvt`, or a distribution mode on the load/store, or a bit-manipulation idiom |
| sub-byte packing | buffer dtype | sometimes free, sometimes a real `vintlv` (see `per_channel/02`) |
| hazards | implicit for registers; `T.sync_threads()` for SMEM | `S.mem_bar("VST_VLD")` by hand, and omitting it silently reads stale data |

The last row is the one that costs correctness rather than effort. Within a vector
scope the hardware does **not** track whether a vector load aliases an earlier
vector store to the same UB address.

## Where the NPU is *easier*: `per_channel`

This is worth knowing because it contradicts the general pattern.

`per_channel` reduces along M — one scale per channel, shared by 32 tokens.

- **On a GPU** consecutive threads hold consecutive *channels*, so reducing along M
  means combining values held by **different threads**. `per_channel_cast_cuda.py`
  abandons `T.Parallel` entirely: it stages partial maxima in shared memory, calls
  `T.sync_threads()`, then has one owner thread per channel combine them, with
  comments about avoiding bank conflicts. It is the only kernel in the quant family
  written that way.
- **On this NPU** 64 channels sit in 64 float32 lanes, one per lane. Reducing 32
  tokens is 32 element-wise `vmax` operations between whole registers. No
  cross-lane reduction, and the resulting scale vector is already laid out one
  value per lane, so applying it is a plain contiguous load — no broadcast either.

The general lesson, and the reason this is in the ladder: **whether a reduction
axis is cheap depends on which axis maps onto the hardware's parallel dimension**,
and lanes-in-a-register and threads-in-a-warp give different answers. Checking
that before choosing a layout is worth more than any individual instruction trick.

A mirror-image case is `per_token/05`, producing column-major scales. On a GPU that
is an index change, possibly with a shared-memory staging step. On the NPU a
register lane cannot move, so it needs `vci` to get per-lane indices and a
`vgather`. And `per_block/04` — the same config on a kernel that produces one
scalar per tile — is free on *both*, because a single value has no layout.

## What does not differ

The *maths* is identical, which is the point of having a torch tier at all:
`puzzles/torch/quant/answer/` implements all 22 variants in plain PyTorch, and
every NPU kernel is checked against it. If a torch variant is three lines and the
NPU variant is eighty, the eighty are all hardware, not algorithm — and reading the
pair makes that distinction concrete rather than rhetorical.

The *schedule* is also shared between the two NPU backends: tiling, UB allocation,
DMA, buffer versions and core assignment are written once and the ASC/VMI choice
only changes what goes inside `T.SimdVF()`. See `pto-vs-asc.md`.
