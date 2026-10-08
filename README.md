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
python -m harness.check asc/cast_back/01 --role puzzle          # prints TODO
```

Variant files are not executable on their own -- they hold the kernel and nothing
else. The harness supplies the shapes, the oracle and the assertions, which is why
a variant file is ~60 lines rather than ~230.

Reading order: `doc/tiers/torch.md` -> `doc/tiers/asc.md` -> `doc/tiers/pto.md`,
then `doc/quant/README.md` for the maths all four kernels share, then each
kernel's `doc/quant/<kernel>/README.md` and its variant pages. Every
variant page carries the algorithm, a worked example, the ASC and PTO code with a
`PTO vs ASC` section, and what the harness checks -- so the prose sits next to the
comparison it is making rather than inside a docstring.

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
python -m harness.check                  # all 69 variants
python -m harness.check torch            # one tier, ~1 min
python -m harness.check asc              # ~11 min
python -m harness.check per_token        # one kernel, all three tiers
python -m harness.check asc/per_token/05 # one variant, streaming
python -m harness.check pto_05           # shorthand: tier + variant number
python -m harness.check --role puzzle    # check the unsolved puzzles
python -m harness.check --list           # list ids without running them
```

Knobs: `TLP_SIM_M`, `TLP_SIM_K`, `TLP_SIM_SOC`, `TLP_SIMULATOR`,
`TLP_CPU_ONLY=1`. See `harness/sim.py`.

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

Cursor and VS Code can open that same image as a dev container from
[`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json). The repo
root is bind-mounted at the image workdir, `/workspace`, so the editor shows
the whole tree.

The same mount from the shell:

```bash
docker run --rm -it -v "$PWD":/workspace tilelang-ascend-puzzles bash
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

| variant | torch | ASC | PTO |
|---|---|---|---|
| `cast_back/01_e4m3_fp32sf` | 4.8 s | 28.3 s | 28.3 s |
| `cast_back/02_fp32_out` | 4.9 s | 28.5 s | 28.5 s |
| `cast_back/03_packed_ue8m0` | 4.9 s | 28.4 s | 27.5 s |
| `cast_back/04_block_sf` | 5.0 s | 27.5 s | 28.5 s |
| `cast_back/05_per_channel_sf` | 4.9 s | 28.5 s | 27.6 s |
| `cast_back/06_fp4_e2m1` | 4.8 s | 29.4 s | 28.5 s |
| `cast_back/07_col_major_compose` | 4.9 s | 29.5 s | 28.5 s |
| `per_token/01_raw_fp32sf` | 4.9 s | 32.7 s | 30.5 s |
| `per_token/02_round_sf` | 4.9 s | 29.6 s | 30.8 s |
| `per_token/03_packed_ue8m0` | 6.3 s | 30.9 s | 29.8 s |
| `per_token/04_fp32_in_fp4_out` | 4.9 s | 31.6 s | 30.1 s |
| `per_token/05_col_major_sf` | 4.9 s | 29.5 s | 30.6 s |
| `per_token/06_split_requant` | 4.9 s | **64.3 s** | **61.2 s** |
| `per_token/07_bf16_fast_compose` | 4.9 s | 30.5 s | 30.6 s |
| `per_block/01_raw_32x32` | 4.9 s | 20.1 s | 20.2 s |
| `per_block/02_round_packed` | 4.9 s | 20.2 s | 20.2 s |
| `per_block/03_fp4_e2m1` | 4.9 s | 21.3 s | 19.3 s |
| `per_block/04_col_major_tma` | 4.9 s | 20.1 s | 20.3 s |
| `per_block/05_split_compose` | 5.0 s | 30.5 s | 30.5 s |
| `per_channel/01_raw_32tokens` | 4.9 s | 18.1 s | 17.2 s |
| `per_channel/02_round_packed_m` | 4.9 s | 19.2 s | 19.3 s |
| `per_channel/03_requant_bf16` | 5.0 s | 19.3 s | 18.2 s |
| `per_channel/04_compose` | 4.9 s | 19.2 s | 19.2 s |
| **whole tier** | **1.9 min** | **10.6 min** | **10.4 min** |

Everything is inside the 60 s per-variant target except `per_token/06`, which
compiles and runs four kernel modes (`full`, `sf_only`, `cast_only`, `requant`) in
one file; it is well inside the 180 s ceiling.

`python -m harness.check --role puzzle` sweeps all 69 unsolved templates in
**6.3 min** (69 TODO). An unimplemented variant costs ~5.7 s rather than ~30 s,
because the harness traces the kernel on the host and short-circuits before
launching the simulator — so working the puzzles does not mean waiting on a
simulator for an answer you have not written yet.

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
doc/                            all prose lives here, as rendered markdown
  tiers/{torch,asc,pto}.md      what each tier is for, and how to read it
  quant/<kernel>/README.md      the kernel: maths, variant index, GPU vs NPU
  quant/<kernel>/NN_name.md     one page per variant, shared by all three tiers
  gpu-vs-npu.md                 the GPU comparison, consolidated
  pto-vs-asc.md                 the ASC/VMI comparison, with measurements
  vf-lane-widths-and-limits.md  register geometry; why 32 is not a legal width
  known-issues.md               toolchain limits, each with a repro script
puzzles/<tier>/quant/           code only -- no prose, no tests, no main
  answer/<kernel>/NN_name.py    the kernel body and its host-side launch
  puzzle/<kernel>/NN_name.py    generated from the answer; implementation removed
harness/                        everything that is not a kernel
  check.py                      sweep a tier, kernel or variant; print a status table
  spec.py                       one Variant record per variant, all three tiers
  variants/<kernel>.py          the per-variant checks
  doc_examples.py               re-verifies the numbers published in doc/quant/
  oracle.py                     the torch reference every tier is compared against
  asserts.py                    FP8 / ULP / byte-equality assertions
  sim.py                        simulator launcher and the re-exec into $HOME
  status.py                     the one contractual [status] line per variant
  consts.py math_ops.py device.py demo.py
  probe/                        reproducible toolchain-limit probes
tools/
  make_puzzles.py               regenerate puzzle/ from answer/ (--check in CI)
  vf_lines.py                   measure VF body size and operation counts
  check_math.py                 lint every .md for GitHub LaTeX pitfalls
docker/                         the container this was validated in
third_party/tilelang            submodule, pinned to the validated commit
```

## How this repo keeps itself honest

A quantization ladder has two easy ways to lie: report `PASS` for a variant that
quietly computed its answer in PyTorch on the host, and ship a status table
describing a toolchain that has since moved. Three mechanisms exist so neither can
happen here:

1. **`harness/status.py` asserts results come back from the device.** A kernel
   variant either runs on the NPU or reports `XFAIL`; `launch()` is never allowed
   to return a host-computed answer. `harness/check.py` tallies `PASS`/`XFAIL`/`TODO`/
   `FAIL` separately and exits nonzero only on `FAIL`, so neither a documented
   toolchain limitation nor an unsolved exercise can masquerade as success.
2. **`harness/sim.py` requires an explicit verdict.** The simulator SIGSEGVs during
   teardown after the program exits; that specific case is tolerated, but a run
   that produced no `PASS`/`XFAIL` at all is a failure, not a pass.
3. **Every documented toolchain limit has a script.** `harness/probe/` reproduces
   each one, tagged with the version measured. Prose goes stale silently; a script
   does not.

Puzzle files are generated from the answers by `tools/make_puzzles.py`, so the
docstring, worked example and tests are literally the same text in both and
`--check` fails if they drift.

## These results are toolchain-specific

Every number and status in this README was measured on the versions pinned above.
Which VF bodies lower is a property of the **toolchain**, not of the hardware, and
it changes between releases in both directions — a body that fails today may lower
after an upgrade, and one that works today is not guaranteed to keep working.

So the limits are not written down as facts. Each one is a script:

```bash
python harness/probe/vf_lane_limits.py          # which VF bodies lower
python harness/probe/fp8_out_idx.py             # whether out_idx can allocate FP8
```

`vf_lane_limits.py` checks four bodies and prints what each one does. On the pinned
versions three lower and one does not — an 8-lane bfloat16→float32 convert fails
with `VMI-LAYOUT-CONTRACT`, which is why `per_block` reduces over a flattened tile
at 64 or 128 lanes rather than following the tile's 32-wide geometry. If that probe
ever reports it lowering, `per_block` can be simplified and
`doc/vf-lane-widths-and-limits.md` needs updating.

`fp8_out_idx.py` prints `out_idx float8 support: NO` today, which is why the
quantize kernels allocate their own outputs. It will print `YES` when that is
fixed.

After upgrading tilelang, ptoas or CANN: run both probes, then
`python -m harness.check` for each tier. `doc/known-issues.md` lists every limit with its
verbatim diagnostic and the probe that reproduces it.

## References

All external material is cited by public URL so this repo stands alone. Nothing
here depends on a sibling checkout.

| what | where | pinned at |
|---|---|---|
| the production quant kernels (`tile_kernels/quant/*_asc.py`, `*_cuda.py`) that this ladder's endpoint is drawn from | [deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels) | — |
| their PTO/VMI port — the diff this repo's PTO-vs-ASC argument rests on | [PTO-ISA/TileKernels-PTO](https://github.com/PTO-ISA/TileKernels-PTO), commit `5395526` on branch `pto-demo` | `5395526` |
| the tilelang fork with the Ascend / PTO backends | [PTO-ISA/tilelang](https://github.com/PTO-ISA/tilelang), branch `pto-dev` | `3d70ede` (this repo's `third_party/tilelang`) |
| the PTO instruction specifications (`docs/PTO-micro-Instruction-SPEC.md`, `docs/PTO-vmi-Instruction-SPEC.md`) | [PTO-ISA/PTO-Gym](https://github.com/PTO-ISA/PTO-Gym) | — |

When a kernel docstring names a file like `per_token_cast_asc.py` without further
qualification, it means `tile_kernels/quant/per_token_cast_asc.py` in TileKernels
(or its PTO port, where the context is VMI).
