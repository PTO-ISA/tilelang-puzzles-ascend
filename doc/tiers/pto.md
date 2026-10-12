# PTO tier — logical VMI

The same 22 variants again, written against the PTO VMI vector IR (`T.vmi`,
imported as `V`). Same chip, same schedules, same results — a different vector
instruction set.

**You can start here.** This tier does not depend on the ASC one: every puzzle
hint is self-contained VMI, and the checks run against `harness/oracle.py`, a
standalone reference. For some people this is the better entry point — `group=`,
`size=` and explicit `vload` + `vbrc` say what you want, where ASC's `BRC_B32` and
`"float32x64"` require knowing what the machine provides.

What *is* ordered is the comparison: each variant page carries a **PTO vs ASC**
section naming what changed and why, and those read better with at least one ASC
kernel behind you. They are commentary, not instructions — skipping them costs you
the argument, not the kernel.

## Running one

```bash
python -m harness.check pto/per_token/01
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
| `per_token/01` | 4 masked reduces + 4 single-element stores → one `vcmax(..., group=4)`; 4 broadcast loads + 2 selects → grouped `vload` slots + `vbrc(..., group=4)`, fused by PTOAS |
| `per_token/02` | the scale computation becomes a **reusable macro** parameterised by lane count, which ASC structurally cannot write |
| `per_token/07` | production's seven-operation deinterleave-pair-regroup dance for bfloat16 group maxima → `vcmax(..., group=8)` |
| `cast_back/05` | FP4's 128 values stop being split into two 64-lane registers |

The broadcast-load source spelling is deliberately two operations. TileLang's
`tilelang/ascend/language/vmi.py` emits a `group_slot_load` for the grouped
`vload`, followed by `group_broadcast` for `vbrc`. PTOAS 0.70 runs
`fuseGroupSlotBroadcastLoads` in `VMIPreAssignmentCombine.cpp` before layout
assignment, producing one `group_broadcast_load`; unit-stride b16/b32 cases lower
to a fused `vlds` `E2B_B16`/`E2B_B32` when layout assignment permits it, or to the
generic BRC-plus-select fallback otherwise.

The API split itself is free in the measured kernels. On the previous toolchain,
all 13 affected references produced exactly the same VF cycles before and after
the source migration, including call counts. On the upgraded toolchain, 12 remain
within -11.1% to +0.6% of their previous cycle counts; these are broader compiler
changes, not a cost of spelling two API calls. The exception is
`per_channel/03`, at 721 → 1,292 cycles: its old trace used `E2B_B32`, while PTOAS
0.70 selects four `BRC_B32` loads plus two `VSEL` operations. That 79.2% regression
is recorded rather than attributed to the API split.

`per_token/01` measures at 39 ASC vector operations against 19 for PTO.

## And where it buys nothing

Stated plainly, because it makes the rest trustworthy. Run:

```bash
python tools/vf_lines.py
```

`cast_back/02`, `cast_back/04`, `cast_back/06`, `per_token/05` and essentially all
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
`doc/tiers/simt.md` is the fourth tier — threads instead of lanes, with the
measured cycle cost in `doc/quant_vf_cycle_comparison.md`;
`doc/gpu-vs-npu.md` steps back to the GPU.
