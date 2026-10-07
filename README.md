# tilelang Ascend quantization puzzles

A guided ladder for four NPU quantization kernels — `cast_back`, `per_token_cast`,
`per_block_cast`, `per_channel_cast` — built three times over:

```
puzzles/torch/quant/   plain PyTorch, on the CPU      <- the prerequisite
puzzles/asc/quant/     Ascend SIMD  (T.simd, "S")     <- the hardware
puzzles/pto/quant/     PTO VMI      (T.vmi,  "V")     <- the comparison
```

**23 variants per tier, same numbering throughout**, so `per_token/03` is the same
feature in all three and the files are directly diffable. Each variant adds exactly
one production config.

Every kernel runs on the device and is checked against the torch tier. Nothing
computes its answer on the host.

## Start here

```bash
# 1. the prerequisite: plain PyTorch, instant, no simulator
python puzzles/torch/quant/answer/cast_back/01_e4m3_fp32sf.py

# 2. the same kernel on the NPU (re-launches itself under the CPU simulator)
python puzzles/asc/quant/answer/cast_back/01_e4m3_fp32sf.py

# 3. the same kernel in VMI
python puzzles/pto/quant/answer/cast_back/01_e4m3_fp32sf.py
```

Then work the puzzle versions, which are the answer files with the implementation
removed and the hint left in its place:

```bash
python puzzles/asc/quant/puzzle/cast_back/01_e4m3_fp32sf.py     # prints TODO
```

Reading order: `puzzles/torch/quant/doc/0.overview.md` →
`puzzles/asc/quant/doc/0.overview.md` → `puzzles/pto/quant/doc/0.overview.md`.

## The ladder

`cast_back` first: it is the only kernel with no reduction (its scale factors are
an input), so it teaches the vector data path before the reduce/broadcast
machinery.

| | cast_back | per_token | per_block | per_channel |
|---|---|---|---|---|
| 01 | FP8 + FP32 scale | raw FP32 scale | raw, 32×32 tile | raw, reduce along M |
| 02 | float32 output | `round_sf` (pow2) | `round_sf` + packed | packed along **M** |
| 03 | packed UE8M0 | packed UE8M0 | FP4 output | requant + bf16 |
| 04 | `sf_block=(32,32)` | fp32 in / FP4 out | column-major scales | composed |
| 05 | `sf_block=(32,1)` | column-major scales | split + composed | |
| 06 | FP4 input | `sf_only`/`cast_only`, requant | | |
| 07 | col-major, composed | bf16 compute, composed | | |

Every production config is covered: `round_sf`, `use_packed_ue8m0` (both pack
axes), `use_tma_aligned_col_major_sf`, FP4 e2m1, float32 input, `with_sf` requant,
`sf_only` / `cast_only`, and the bfloat16 compute path.

The endpoint is the SIMD-VF part of
[TileKernels](https://github.com/deepseek-ai/TileKernels)' quant kernels and their
PTO port. What is deliberately **out of scope**: multi-core `T.Persistent`
scheduling, double-buffered UB, and L2 cache hints. Those change no number any
kernel here computes, and they would make a cycle-accurate simulation far slower
without teaching anything about the vector unit. Single core, `T.Kernel(1)`,
throughout.

## Environment

No NPU is required — and none is present in the container. Each NPU variant
re-launches itself under `msprof op simulator` (SoC `Ascend950PR_9599`), a
cycle-accurate CPU model.

```bash
python run_all.py --tier torch          # CPU, ~2 min
python run_all.py --tier asc            # ~11 min
python run_all.py --tier pto            # ~11 min
python run_all.py --kernel per_token    # one kernel, all three tiers
python run_all.py --role puzzle         # check the unsolved puzzles
```

Knobs: `TLP_SIM_M`, `TLP_SIM_K`, `TLP_SIM_SOC`, `TLP_SIMULATOR`,
`TLP_CPU_ONLY=1`. See `common/sim.py`.

`cannsim` / `npusim` is selectable with `TLP_SIMULATOR=cannsim` but **hangs on
`vshr`/`vshl`**, which every variant from `per_token/02` onward needs, so it
cannot run most of the ladder. That is why msprof is the default.

### Building the container

The image is what everything above was validated in. The build copies the
vendored tilelang into `/opt/tilelang` and installs it from source, so the
submodule has to be checked out first:

```bash
git submodule update --init --recursive        # pins tilelang at 3d70ede
docker build -f docker/Dockerfile -t tilelang-ascend-puzzles .
```

Base image `quay.io/ascend/cann:9.2.0-beta.2-950-ubuntu22.04-py3.12`. Inside the
container, `tilelang-smoke` runs upstream tilelang's own Ascend test selection as
an independent check that the toolchain itself is working, separately from this
repo's ladder.

Versions this repo was validated against, 2026-10-07:
tilelang **0.1.15** (submodule `third_party/tilelang` @ `3d70ede`, branch
`pto-dev`), ptoas vmi **0.1.9**, CANN **9.2.0-beta.2**, torch 2.9.0+cpu,
torch_npu 2.9.0.post6, Python 3.12.

## Measured status

All 23 variants pass on all three tiers. Shapes are `M=32, K=128` by default;
`per_token/07` uses `K=256` (the bfloat16 path steps 256 values) and the
packed-along-M `per_channel` variants use `M=64` (packing needs an even number of
token groups).

Simulator wall time, measured end to end including compilation:

| variant | ASC | PTO |
|---|---:|---:|
| `cast_back/01_e4m3_fp32sf` | 28.2 s | 28.3 s |
| `cast_back/02_fp32_out` | 28.3 s | 28.3 s |
| `cast_back/03_packed_ue8m0` | 27.3 s | 28.3 s |
| `cast_back/04_block_sf` | 28.4 s | 28.2 s |
| `cast_back/05_per_channel_sf` | 27.1 s | 28.3 s |
| `cast_back/06_fp4_e2m1` | 27.3 s | 29.2 s |
| `cast_back/07_col_major_compose` | 29.3 s | 27.1 s |
| `per_token/01_raw_fp32sf` | 31.4 s | 29.3 s |
| `per_token/02_round_sf` | 29.4 s | 31.5 s |
| `per_token/03_packed_ue8m0` | 29.3 s | 30.4 s |
| `per_token/04_fp32_in_fp4_out` | 32.5 s | 29.4 s |
| `per_token/05_col_major_sf` | 30.5 s | 30.4 s |
| `per_token/06_split_requant` | **63.1 s** | **62.1 s** |
| `per_token/07_bf16_fast_compose` | 31.5 s | 30.3 s |
| `per_block/01_raw_32x32` | 20.0 s | 20.1 s |
| `per_block/02_round_packed` | 20.1 s | 20.1 s |
| `per_block/03_fp4_e2m1` | 19.0 s | 19.0 s |
| `per_block/04_col_major_tma` | 20.1 s | 20.0 s |
| `per_block/05_split_compose` | 30.4 s | 31.4 s |
| `per_channel/01_raw_32tokens` | 19.0 s | 17.9 s |
| `per_channel/02_round_packed_m` | 20.2 s | 19.0 s |
| `per_channel/03_requant_bf16` | 19.0 s | 19.0 s |
| `per_channel/04_compose` | 19.0 s | 19.0 s |
| **whole tier** | **10.5 min** | **10.4 min** |

Everything is inside the 60 s per-variant target except `per_token/06`, which
compiles and runs four kernel modes (`full`, `sf_only`, `cast_only`, `requant`) in
one file; it is well inside the 180 s ceiling.

### Numerical fidelity

Scale factors are bit-exact against torch everywhere. FP8 and FP4 values are
byte-exact everywhere **except `per_channel/03`**, which is the one deliberate
exception and is asserted rather than tolerated: requantizing input that already
sits on the FP8 grid produces *exact ties* (e.g. dequantized 4.5 with scale 6/448
is exactly 336.0, precisely halfway between e4m3's 320 and 352). 97 of 4096 values
land on such ties and the hardware and torch break them differently. The test
asserts that every difference is at most one code **and** that every differing
position is a genuine midpoint, so it still fails on a real defect.

## PTO vs ASC, measured

```bash
python tools/vf_lines.py
```

Across the 23 paired variants, PTO uses about **three quarters** of ASC's static
vector-operation count — concentrated, not uniform:

- **Large wins**: all of `per_token` (`group=` replacing mask-and-select — variant
  01 is 39 operations against 19), `cast_back/06` and `per_token/07` (vector width,
  FP4 and bfloat16 paths).
- **Draws**: `cast_back/03`, `/05`, `/07`, `per_token/05`, and essentially all of
  `per_block` and `per_channel`.
- **PTO slightly longer**: several `per_block` variants, because VMI requires
  explicit `size=` and mask operands and there is nothing to factor out.

The rule that explains the pattern: **VMI pays where ASC had to emulate something
the hardware does not directly offer.** `doc/pto-vs-asc.md` has the detail,
including VMI's genuine regressions (no per-lane variable shift, no scalar-operand
`vmul`/`vsub`, reversed `vsel` argument order).

Two caveats stated in the tool's own output: the counts are *static*, so PTO's
wider vectors running fewer loop iterations do not show up; and source *lines* run
about 90% where operations run 74%, which is the ergonomic price of VMI's
explicitness.

## Layout

```
README.md
doc/
  gpu-vs-npu.md                 the GPU comparison, consolidated
  pto-vs-asc.md                 the ASC/VMI comparison, with measurements
  vf-lane-widths-and-limits.md  register geometry; why 32 is not a legal width
  known-issues.md               toolchain limits, each with a repro script
puzzles/<tier>/quant/
  answer/<kernel>/NN_name.py    complete, documented, runnable
  puzzle/<kernel>/NN_name.py    generated from the answer; implementation removed
  doc/                          tier overview + per-kernel concept
common/                         harness: oracle, checks, simulator launcher
  probe/                        reproducible toolchain-limit probes
tools/
  make_puzzles.py               regenerate puzzle/ from answer/ (--check in CI)
  vf_lines.py                   measure VF body size and operation counts
run_all.py                      sweep a tier or kernel, print a status table
docker/                         the container this was validated in
third_party/tilelang            submodule, pinned to the validated commit
```

## How this repo keeps itself honest

The predecessor to this repo printed `PASS` for variants that had quietly fallen
back to computing the answer in PyTorch on the host, and shipped a pass/fail table
describing a toolchain two releases old. Three mechanisms exist to prevent that:

1. **`common/status.py` asserts results come back from the device.** A kernel
   variant either runs on the NPU or reports `XFAIL`; `launch()` is never allowed
   to return a host-computed answer. `run_all.py` tallies `PASS`/`XFAIL`/`TODO`/
   `FAIL` separately and exits nonzero only on `FAIL`, so neither a documented
   toolchain limitation nor an unsolved exercise can masquerade as success.
2. **`common/sim.py` requires an explicit verdict.** The simulator SIGSEGVs during
   teardown after the program exits; that specific case is tolerated, but a run
   that produced no `PASS`/`XFAIL` at all is a failure, not a pass.
3. **Every documented toolchain limit has a script.** `common/probe/` reproduces
   each one, tagged with the version measured. Prose goes stale silently; a script
   does not.

Puzzle files are generated from the answers by `tools/make_puzzles.py`, so the
docstring, worked example and tests are literally the same text in both and
`--check` fails if they drift.

## Reading the predecessor's conclusions with care

[`tilelang-puzzles-ascend`](https://github.com/learning-chip/tilelang-puzzles-ascend) documented that a fused 128-lane per_token body could not
compile (`VMI-UNSUPPORTED` on `pto.vmi.group_broadcast`) and fell back to host
torch for every `per_block` variant and for `per_token` float32-input
(`VMI-RESIDUAL-OP`), against ptoas 0.1.8.

**All of those work on 0.1.9.** The blocker was a formulation problem, not a
toolchain wall. One real limit survives — an 8-lane bfloat16→float32 convert fails
with `VMI-LAYOUT-CONTRACT` — and it is why `per_block` reduces over a flattened
tile at 64 or 128 lanes instead of following the tile's 32-wide geometry.

`common/probe/vf_lane_limits.py` re-checks all four bodies on every run.

## References

All external material is cited by public URL so this repo stands alone. Nothing
here depends on a sibling checkout.

| what | where | pinned at |
|---|---|---|
| the production quant kernels (`tile_kernels/quant/*_asc.py`, `*_cuda.py`) that this ladder's endpoint is drawn from | [deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels) | — |
| their PTO/VMI port — the diff this repo's PTO-vs-ASC argument rests on | [PTO-ISA/TileKernels-PTO](https://github.com/PTO-ISA/TileKernels-PTO), commit `5395526` on branch `pto-demo` | `5395526` |
| the tilelang fork with the Ascend / PTO backends | [PTO-ISA/tilelang](https://github.com/PTO-ISA/tilelang), branch `pto-dev` | `3d70ede` (this repo's `third_party/tilelang`) |
| the PTO instruction specifications (`docs/PTO-micro-Instruction-SPEC.md`, `docs/PTO-vmi-Instruction-SPEC.md`) | [PTO-ISA/PTO-Gym](https://github.com/PTO-ISA/PTO-Gym) | — |
| this repo's predecessor, whose conclusions are revisited above | [learning-chip/tilelang-puzzles-ascend](https://github.com/learning-chip/tilelang-puzzles-ascend) | — |

When a kernel docstring names a file like `per_token_cast_asc.py` without further
qualification, it means `tile_kernels/quant/per_token_cast_asc.py` in TileKernels
(or its PTO port, where the context is VMI).
