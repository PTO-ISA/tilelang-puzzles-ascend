# PTO tier — logical VMI

The same 23 variants again, written against the PTO VMI vector IR (`T.vmi`,
imported as `V`). Same chip, same schedules, same results — a different vector
instruction set.

Read the ASC tier first. These files are written as a comparison with it, and each
one carries a **PTO vs ASC** section naming what changed and why.

## Running one

```bash
python puzzles/pto/quant/answer/per_token/01_raw_fp32sf.py
```

Identical harness to the ASC tier. The only difference in the source is the import
and the jit target:

```python
from tilelang.ascend.language import vmi as V     # instead of simd as S
@tilelang.jit(target="pto")                       # instead of target="ascend"
```

## What to look for

VMI's whole contribution is that **width and segment count become arguments**
instead of being baked into instruction names and dtype strings. Four places in
this ladder show it clearly:

| file | what collapses |
|---|---|
| `per_token/01` | 4 masked reduces + 4 single-element stores → one `vcmax(..., group=4)`; 4 broadcast loads + 2 selects → one `vload(..., dist_mode="brc", group=4)` |
| `per_token/02` | the scale computation becomes a **reusable macro** parameterised by lane count, which ASC structurally cannot write |
| `per_token/07` | production's seven-operation deinterleave-pair-regroup dance for bfloat16 group maxima → `vcmax(..., group=8)` |
| `cast_back/06` | FP4's 128 values stop being split into two 64-lane registers |

`per_token/01` measures at 39 ASC vector operations against 19 for PTO.

## And where it buys nothing

Stated plainly, because it makes the rest trustworthy. Run:

```bash
python tools/vf_lines.py
```

`cast_back/03`, `cast_back/05`, `cast_back/07`, `per_token/05` and essentially all
of `per_block` and `per_channel` are draws, and several `per_block` variants come
out *longer* in PTO because VMI requires explicit `size=` and mask operands.

The rule that explains the pattern, and the thing actually worth learning:

> VMI pays where ASC had to **emulate** something the hardware does not directly
> offer — a segment width it lacks, a conversion expressed as a distribution mode,
> a vector wider than one register. It pays nothing where ASC was already saying
> exactly what it meant.

`per_channel/03` is the cleanest demonstration: within one kernel, the input scale
varies along K (needs a broadcast → VMI wins) and the output scale varies along M
(one value per lane already → no difference).

## Genuine regressions

- **No per-lane variable shift.** `V.vshrs` takes a scalar amount only, where
  `S.vshr` accepts a vector. Production's PTO `cast_back` has to compute both byte
  positions and select between them for something ASC does in one shift.
- **No scalar-operand `vmul` / `vsub`.** Constants must be broadcast first.
- **`vsel` argument order is reversed** — `V.vsel(mask, a, b)` against
  `S.vsel(a, b, mask)`. A porting hazard that produces wrong numbers, not an error.

## The schedule is not VMI's business

Everything outside `with T.SimdVF():` — tiling, UB allocation, buffer versions,
DMA, core assignment — is identical between the two tiers. That is not an accident
of how this repo was written: production's PTO port
([TileKernels-PTO](https://github.com/PTO-ISA/TileKernels-PTO) commit `5395526`) touched *only* the vector bodies across all
four kernels, leaving every `@T.prim_func` byte-identical, for a net 83 lines
removed. `per_block/04` is the variant where this is most visible: it changes the
loop nest and the DMA, and the two backends' files are the same below the
docstring.

## A note on citations

Where these files name a production kernel like `per_token_cast_asc.py`, that is
`tile_kernels/quant/per_token_cast_asc.py` in
[deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels), or in its
PTO port [PTO-ISA/TileKernels-PTO](https://github.com/PTO-ISA/TileKernels-PTO)
where the context is VMI. The instruction specifications are in
[PTO-ISA/PTO-Gym](https://github.com/PTO-ISA/PTO-Gym) under `docs/`. See the
README's References table. Nothing in this repo needs a sibling checkout.

## Next

`doc/pto-vs-asc.md` consolidates the comparison with the measurements;
`doc/gpu-vs-npu.md` steps back to the GPU.
