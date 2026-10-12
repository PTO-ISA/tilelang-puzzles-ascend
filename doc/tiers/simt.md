# SIMT tier — the thread model

The same 22 variants a fourth time, written as a GPU programmer would write them
on day one: `with T.SimtVF(threads=128):`, `T.alloc_fragment`, `T.Parallel`, and
plain arithmetic on scalars. No lane counts, no masks, no distribution modes.

**You can start here**, though it is not the obvious choice. Everything is
self-contained and the checks run against `harness/oracle.py` like every other
tier. But the comparison sections on each variant page are written as *SIMT
against the SIMD tiers*, so they read better with one ASC or PTO kernel behind
you. The algorithm itself belongs to the [torch tier](torch.md) — do that first.

## Running one

```bash
python -m harness.check simt/per_token/01
```

The source differs from the SIMD tiers in two lines:

```python
import tilelang.ascend.language as T       # no `simd as S`, no `vmi as V`
@tilelang.jit(target="ascend")             # NOT target="pto"
```

**The target is `ascend`, not `pto`.** That is not a typo or a stylistic
preference: on the pinned toolchain the upstream SIMT suite passes 56 of 61 cases
against the Ascend backend and 33 of 61 against PTO. SIMT is an Ascend-backend
style; it does not pair with PTO the way ASC and PTO pair with each other.

## The shape of every kernel here

```python
with T.Kernel(1):
    x_ub = T.alloc_shared((hidden,), T.bfloat16)      # stage through UB
    T.copy(X[token, 0], x_ub)
    with T.SimtVF(threads=128):
        q_l = T.alloc_fragment((hidden,), T.float32)  # per-thread registers
        for i in T.Parallel(hidden):                  # one thread per element
            q_l[i] = ...
        T.copy(q_l, q_ub)                             # converts on the way out
    T.copy(q_ub, Q[token, 0])
```

`T.copy` between buffers of different dtypes **converts**, which is why the
`out_dtype` switches that the SIMD tiers spell as store modes or `vcvt` calls
need no code here at all: the declared dtype of the destination carries the
configuration.

## The one rule that decides what SIMT code looks like

> Inside a single `T.Parallel` body, a **fragment** may be touched with exactly
> one structurally identical index expression. Shared and UB buffers are exempt.

That is `RecordBufferAccess` in `src/op/parallel.cc`, and nearly every design
choice in this tier follows from it. Irregular, packed or data-dependent index
math lives on `alloc_shared`; fragments carry only the clean subscripts of the
`Parallel` variables. When a kernel here stages intermediate values in a shared
scratch buffer rather than a fragment — `per_token/06`, `per_channel/03`,
`per_block`'s partials — that is why.

## Two traps that cost real debugging time

**A 2-D UB staging buffer silently reads the wrong rows.**
`T.copy(X[token, 0], x_ub)` into a `(num_groups, group_size)` buffer copies the
2-D *region* `X[token:token+num_groups, 0:group_size]` — one group's worth from
each of several **tokens** — not the token's contiguous channels. Group 0 comes
out right and the rest are other tokens' data, which looks exactly like a broken
reduction. Keep UB buffers flat and put the structure in the index.

**A `T.Parallel` body may not touch one fragment two ways.** The error is
`Check failed: StructuralEqual()(it->second.indices, indices)`, naming the two
expressions — `(p * 2 + 1,)` and `(p * 2,)` — which is clear once you have seen
it and baffling the first time. Shared buffers are exempt, so staging there is
the fix.

**`T.abs` does not lower on bfloat16, and nor does a bfloat16 accumulator.**
`tirx.fabs` is unresolved with the abs, and without it the Ascend device
compile fails outright. Reduce in float32 and convert after; this is why
`per_token/07` and `per_channel/04`, the two variants whose whole point is
bfloat16 lane width, have float32 SIMT bodies.

## Where SIMT wins

| file | what disappears |
|---|---|
| `cast_back/01` | the group broadcast — ASC's two `BRC_B32` loads and a `vsel`, PTO's `dist_mode="brc"` — becomes `sf_ub[i // group_size]` |
| `per_token/05` | the in-register transpose. ASC builds a lane-index vector with `S.vci` and permutes a block through `S.vgather2`; SIMT stores at `sf_out_ub[g, sub]` |
| `per_channel/02` | the M-packing interleave. `V.vintlv` plus a padded scratch row becomes two scalar stores — and a scalar store cannot overrun, so the padding goes too |
| `per_token/02`, `per_block/02` | the exponent bit trick, as scalar integer arithmetic with no width to thread through it |

The pattern worth extracting: **SIMT pays off where the SIMD tiers had to move
data between lanes.** Arithmetic is not where the win is — the vector unit does
arithmetic well, and `doc/pto-vs-asc.md` records that the bit math is nearly free
in both SIMD tiers. The win is irregular data *placement*, where a vector
register's fixed lane assignment is the thing in the way.

## Where it buys nothing, or costs

Stated plainly, because it makes the rest trustworthy.

| file | what happens |
|---|---|
| `cast_back/04` | nothing. The SIMD tiers are getting a refund here (the broadcast disappears when there is one scale per channel); SIMT never paid it, so its line count does not move |
| `per_block/01`, `/04` | a draw. A tile scale is a scalar, so no tier has lane-placement work to do; the two-level reduction has the same shape in all three |
| `per_channel/01` | a draw, and the mirror of why PTO draws here: a channel is a lane for SIMD and a thread for SIMT, so the data already lines up either way |
| `per_token/04`, `per_block/03` | **SIMT is decisively worse.** No FP4 convert is reachable from a `SimtVF` region, so the e2m1 encode is written out by hand: 18 lines against PTO's 4-operation round-to-odd. `per_token/04` ships both a natural and a branch-free spelling of that encoder, 1.78x apart — see below |
| `per_token/07`, `per_channel/04` | the variant's *lesson* does not port. Both exist to show bfloat16 doubling the lane count; SIMT has no lane width to trade, so its body is an earlier variant's at a larger size |
| `cast_back/05` | PTO wins. A format conversion is one thing the hardware does, so the tier that can name it directly should be shortest |

## The measurements

Two tools, two different axes, and they disagree:

```bash
python tools/vf_lines.py        # static: lines per tier, ops for ASC/PTO
python tools/vf_cycles.py --all # measured: simulated VF cycles per tier
```

Across all 22 variants SIMT is **53% of ASC's source lines** inside the VF
region, and it costs **3.20x to 99.62x ASC's simulated VF cycles**. The top of
that range is one special case — `per_token/04`, where an 18-line hand-written
e2m1 encoder stands in for a hardware conversion — and setting it aside the
range is **3.20x to 21.18x, mean 10.44x**. The full per-variant table, with
absolute cycle counts and the method, is in
[`quant_vf_cycle_comparison.md`](../quant_vf_cycle_comparison.md).

Two caveats on the line count, both of which make SIMT look better than it is.
`tools/vf_lines.py` counts only what is inside the VF region, so PTO's `@T.macro`
bodies and SIMT's Python expression builders are excluded in equal measure — but
at `per_token/04` and `per_block/03` PTO's excluded helper is five lines and
SIMT's is eighteen. And there is no `ops` column for SIMT at all: its arithmetic
is plain operators, so the backend picks the instructions and the source has
nothing comparable to count.

## What the fourth tier adds to the story

The three-tier version reads as a line: ASC names the machine, PTO names the
intent, and PTO costs 74% of ASC's vector operations for the same results. It is
tempting to extend that line and call SIMT the far end — easiest to write,
slowest to run.

The measurements do not support reading it as one line.

- **PTO is sometimes faster than ASC, not merely close.** On `per_token/01` PTO
  uses about three quarters of ASC's cycles. "Most verbose is fastest" is not
  what the simulator says.
- **`per_token/05` is where the two axes most clearly separate**: it is
  simultaneously the variant where SIMT's source is shortest relative to ASC's
  and the variant where its cycle ratio is worst. Both facts have the same
  cause — the thread model removes the lane-placement problem, and leaves most of
  the vector unit idle doing it.
- **Short source and fast kernel are not two readings of one axis.** They are two
  axes, and the four tiers are a trade rather than a ranking.

### One variant ships two spellings, 1.78x apart

`per_token/04` is the exception to "one kernel per variant", and the reason is
worth knowing before you write any SIMT code here.

`T.if_then_else` on a data-dependent per-element condition lowers to a **real
branch** on this toolchain, not a predicated select. In that variant's encoder
there are three such conditionals, and the cost is 4.94 cycles per instruction
against the rest of the tier's 1.90–2.01 — it is the only variant in the ladder
whose trace carries a large number of `SIMT_START_DVG` markers. Respelling the
conditionals as mask arithmetic (`m = -T.Cast("int32", cond)`, then
`(a & m) | (b & ~m)`) issues **45% more instructions and runs 1.78x fewer
cycles**: 613,632 down to 345,189, which is 56.0x ASC instead of 99.6x.

Both are in the file behind `branch_free=`, both are checked by the harness to
be byte-identical, and both are measured in
[`quant_vf_cycle_comparison.md`](../quant_vf_cycle_comparison.md). The default is
the conditional, because that is the spelling this ladder set out to measure.

On a GPU the choice would not exist: a compiler predicates that conditional into
a select. So this is a **codegen gap, not a cost of the thread model** — and it
is the one place where writing the natural thing and writing the fast thing
genuinely diverge in this tier. `doc/known-issues.md` records it under
performance cliffs; `harness/probe/simt_divergence.py` is the controlled
experiment.

### And do not over-read the cycle column either

Every variant here has SIMT slower, and the symmetry of that is suspicious
enough to state its limits. These are 22 single-config kernels at one shape,
whose reductions all use `group_size = 32` — the case that leaves the most
threads idle. A separate SIMT-versus-SIMD suite on the same simulator, sweeping
several SIMT strategies per kernel, found ratios from **0.33x to 53x**: SIMT
winning outright on some grouped reductions. And these kernels are deliberately
untuned, which was the comparison asked for; in that suite the strategy choice
alone moved one identical kernel by a factor of two.

So the claim this tier supports is narrow and worth stating precisely: *written
the obvious way, on these kernels, at this shape, the thread model trades cycles
for ease.* Not that SIMT is slow.

What survives as general guidance: SIMT is the fastest way to get a correct
kernel, and a reasonable place to start one. Where it is also fast enough,
stopping there is a real option. Where it is not, the variant pages say which
lane-level operation the SIMD tiers are buying with their extra lines.

## A note on citations

The SIMT style here follows the upstream Ascend SIMT examples and is close to the
CUDA kernels in `tile_kernels/quant` in
[deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels) — `T.Parallel`
is largely the same syntax. The ASC and PTO references are listed on
[`pto.md`](pto.md); nothing in this repo needs a sibling checkout.

## Next

`doc/quant_vf_cycle_comparison.md` for the measured table;
`doc/pto-vs-asc.md` for the two-SIMD-tier comparison;
`doc/gpu-vs-npu.md` steps back to the GPU, and is the natural companion to this
page.
