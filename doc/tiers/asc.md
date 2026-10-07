# ASC tier — Ascend SIMD

The same 23 variants as the torch tier, written against the Ascend SIMD vector IR
(`T.simd`, imported as `S`). These run on the chip — or, here, on a cycle-accurate
CPU model of it.

Do the torch tier first. Every kernel here is checked against it.

## Running one

```bash
python puzzles/asc/quant/answer/cast_back/01_e4m3_fp32sf.py
```

No NPU is present, so the file re-launches itself under `msprof op simulator`
(`Ascend950PR_9599`). Expect 12–27 s for most variants; see the README for the
measured table. An unsolved puzzle reports `TODO` in about 6 s, without paying for
a simulator launch.

Knobs: `TLP_SIM_M`, `TLP_SIM_K`, `TLP_SIMULATOR`, `TLP_CPU_ONLY=1`. See
`harness/sim.py`.

## Read `01` of each kernel properly

The four `01` files are where the hardware model is explained, and later variants
assume it:

- **`cast_back/01`** — the memory hierarchy (GM → UB → vector register and back),
  what `T.alloc_shared` actually is on this backend, why the loop steps by 64, and
  what a distribution mode (`dist=`) is.
- **`per_token/01`** — the three-pass reduce/scale/apply structure and the two
  `S.mem_bar("VST_VLD")` barriers between them.
- **`per_block/01`** — why the tile's own 32-wide geometry is unusable and how to
  reduce across a flattened tile instead.
- **`per_channel/01`** — the one kernel where this hardware has the easy job.

Each file's docstring is its implementation guide: the maths, the instruction-level
reasoning, and a **GPU vs NPU** section. There is no separate guide document,
deliberately — prose that sits next to the code it describes cannot drift away
from it.

## The four things that make this tier different from torch

**1. A vector register is 256 bytes.** 64 float32 lanes, 128 bfloat16, 256 bytes.
Every loop bound in these kernels comes from that number, never from the problem
size. `doc/vf-lane-widths-and-limits.md` has the table and the legal-width list.

**2. The quant group is 32, and 32 is not a legal vector width.** That single
mismatch generates most of the awkwardness in `per_token` and `cast_back`: a
64-lane register straddles two groups, so reductions need masks and scale
broadcasts need a select.

**3. Memory hazards are yours.** Inside a vector scope the hardware does not track
whether a vector load aliases an earlier vector store to the same UB address.
Forget `S.mem_bar("VST_VLD")` and you read stale data, silently.

**4. Distribution modes do the dtype work.** `UNPK4_B8` widens on load, `PK4_B32`
narrows on store, `BRC_B32` broadcasts one scalar across the register. Several
float↔float conversions have no instruction at all and are bit-manipulation idioms
instead — `S.vdintlv` keep-high is a truncating float32→bfloat16; `S.vintlv(zero,
x)` is the widening direction.

## Single core, on purpose

Every kernel is `with T.Kernel(1)`. Production uses `T.Persistent` across 64 vector
cores with double-buffered UB, and that is pure scheduling: it changes no number
any of these kernels computes, and it would make the cycle-accurate simulation
far slower without teaching anything about the vector unit. Where a variant's
schedule *does* matter — `cast_back/04` hoisting the scale DMA, `per_token/05`
batching tokens so the transpose has something to transpose — the file says so.

## Honest status

Every variant in this tier runs on the device and is checked against the torch
oracle. Nothing computes its answer on the host: `harness/status.py` asserts that
results come back from `npu`, so a torch fallback cannot hide behind a `PASS`.

Where a result is *not* bit-exact, the test says so and bounds it rather than
loosening a tolerance — `per_channel/03` is the one such case, and its docstring
derives why (requantization produces exact ties).

Toolchain limits hit while writing these are in `doc/known-issues.md`, each with a
reproducing script in `harness/probe/`.

## A note on citations

Where these files name a production kernel like `per_token_cast_asc.py`, that is
`tile_kernels/quant/per_token_cast_asc.py` in
[deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels), or in its
PTO port [PTO-ISA/TileKernels-PTO](https://github.com/PTO-ISA/TileKernels-PTO)
where the context is VMI. The instruction specifications are in
[PTO-ISA/PTO-Gym](https://github.com/PTO-ISA/PTO-Gym) under `docs/`. See the
README's References table. Nothing in this repo needs a sibling checkout.

## Next

`doc/tiers/pto.md` — the same 23 variants in logical VMI, and
`doc/pto-vs-asc.md` for what that changes.
