# per_block — one scale per 32×32 tile — PTO tier

## The maths

```
    amax = max(|x|) over a 32x32 tile
    sf   = max(amax, 1e-4) / 448                 (optionally rounded up to 2^k)
    q    = cast_e4m3(x * (448 / amax))

    x: (M, K)   q: (M, K)   sf: (M/32, K/32)

The weight quantization granularity: 1024 values per scale, so the scales are
amortised across every token in the batch.
```

Implemented in plain PyTorch in `puzzles/torch/quant/answer/per_block/`, which is
what every kernel here is checked against.

## The variant ladder

| # | config | new idea |
|---|---|---|
| 01 | raw float32 scale | a **2-D** reduction, and why the tile's own width is unusable |
| 02 | `round_sf` + `use_packed_ue8m0` | the same exponent trick on one scale per tile |
| 03 | FP4 output | the coarsest granularity × the coarsest format |
| 04 | column-major scales | **free here**, unlike `per_token/05` |
| 05 | `sf_only` / `cast_only`, composed | `cast_only` is bit-exact with a pow2 scale |

Each variant's own docstring is its implementation guide — the instruction-level
reasoning lives next to the code it describes, where it cannot drift away from it.
Start with `01`.

## GPU vs NPU

The CUDA version uses 256 threads and `T.reduce_absmax` over a reshaped fragment;
the two-dimensional reduction is no harder for it than a one-dimensional one,
because a fragment's shape is just an index space.

On the NPU the shape matters, and this kernel is where the lane-width rule bites
hardest. A 32×32 tile is 32 values wide, which looks like the natural vector width
and is not available: the legal lengths are `{1, 2, 4, 8, 64, 128, 256}` and 32 is
absent (half a register). The obvious fallback of 8 lanes — one 32-byte lane group
— **does not compile** for the needed bfloat16→float32 convert:

```
VMI-LAYOUT-CONTRACT: pto.vmi.extf has no registered layout support
```

(`common/probe/vf_lane_limits.py` reproduces it.) So the kernel flattens the tile
into one 1024-value run and reduces at 64 or 128 lanes, which is also what
production does. The transferable rule: **pick the lane width from the hardware and
reshape the problem to fit it**, not the other way round. A GPU never asks this
question.

Variant 04 is the interesting counterpoint. Column-major scales cost an in-register
gather in `per_token/05` and **nothing at all** here — on either backend, and on a
GPU too. The difference is how many values the producing loop holds at once: one
scalar per tile has no layout; a register-full does.

## PTO vs ASC

**per_block is the kernel where VMI buys the least**, and saying so is part of the
comparison being worth anything.

`tools/vf_lines.py` puts PTO's *static* operation count slightly **higher** than
ASC's on four of five variants. Two reasons, both real:

1. The reduction is a whole-vector `group=1` reduce. The segmented-operation
   advantage that drives `per_token` does not apply.
2. VMI requires explicit `size=` and mask operands, so each call is wider.

The static count also *understates* PTO here, and that cuts the other way: PTO
reduces 128 lanes per iteration against ASC's 64, so it runs **8 iterations rather
than 16** and issues fewer instructions at runtime. A count of operations written
cannot see a count of operations executed.

The honest summary for this kernel: VMI is not shorter, it is wider.

Two smaller differences that do show up:

- Writing the scale byte is `V.vcvt(biased, "uint8")` against a reinterpret to
  `"uint8x256"` plus a matching `PK4_B32` store mode — two things that must agree
  and both of which compile when they do not.
- Reading a given scale byte (variant 05, `cast_only`): ASC's `BRC_B32` broadcasts
  the 32 bits at an address, so a single byte arrives with three neighbours that
  have to be masked off. VMI widens on conversion and respects the element
  boundary. This is the class of difference where a forgotten mask still compiles
  and still produces plausible numbers.

See `doc/pto-vs-asc.md` for the consolidated comparison and
`python tools/vf_lines.py` for the per-variant measurements.

## Further reading

- `doc/vf-lane-widths-and-limits.md` — the register geometry and legal widths
- `doc/known-issues.md` — toolchain limits, each with a reproducing script
- `doc/gpu-vs-npu.md` — the consolidated GPU comparison
