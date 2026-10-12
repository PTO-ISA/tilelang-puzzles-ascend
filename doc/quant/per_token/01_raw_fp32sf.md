# per_token 01 — quantize to FP8 with a raw float32 scale

The first kernel in the ladder with a reduction. Read the
[kernel overview](README.md) for the three-pass structure and the barriers; this
page is about how the reduction itself is written.

## Worked example

One token, $K = 64$, so two groups of 32. Only the first four values are nonzero:

| | value |
|---|---|
| `x[0, 0:4]` | `[1.0, 2.0, -4.0, 0.5]` |
| `amax` for group 0 | `4.0` |
| `sf[0, 0]` | `0.00892857`, i.e. $\frac{4}{448}$ |
| the multiplier | $\frac{448}{4} = 112$ |
| `q[0, 0:4]` | `[112, 224, -448, 56]` |

The largest magnitude lands exactly on $-448$, the e4m3 limit. Nothing saturates,
nothing is wasted — which is what a per-group scale buys.

An all-zero group, for contrast: `amax` clamps to `1e-4`, so `sf` becomes
`2.23e-07` and `q` stays `0` instead of becoming `NaN`.

## Torch

```python
grouped = x.float().view(m, k // group_size, group_size)
amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
sf = amax / E4M3_MAX
q = (grouped * (E4M3_MAX / amax).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
```

## ASC — reducing 32 of 64 lanes

`S.vcmax` reduces across a register and produces one value. But a register holds
64 float32 lanes and a group is 32 channels, so each register contains **two**
groups. The reduction has to be restricted to half the lanes at a time:

```python
mask_low  = S.pset(32, "PAT_VL32")          # lanes 0-31
mask_all  = S.pset(32, "PAT_ALL")
mask_high = S.pnot(mask_low, mask_all)      # lanes 32-63

S.vsts(amax_ub[group],     S.vcmax(abs_x, mask_low),  dist="ONEPT_B32")
S.vsts(amax_ub[group + 1], S.vcmax(abs_x, mask_high), dist="ONEPT_B32")
```

Two masked reductions and two single-element stores (`ONEPT_B32` writes one lane's
worth) per register.

A frontend detail worth knowing: **`S.pset` must be bound to a name** before use.
Nesting it — `S.pnot(mask_low, S.pset(32, "PAT_ALL"))` — leaves
`tl.simd.pset` unresolved at lowering, with an unhelpful `InternalError`.

### Pass 2 is 64 scales at a time

A nice detail: the amax values were written as individual elements, but they are
now *contiguous*, so one 64-lane load picks up 64 of them and the clamp, divide and
reciprocal all happen 64 groups at a time. Writing the reduction's output back to
UB converts a cross-lane problem into an element-wise one.

## PTO vs ASC — the two collapses

### The reduction: eight operations become three

VMI's reduce takes a `group=` argument meaning "this vector is $N$ independent
segments; reduce each one". One 128-lane vector, four segments, four results
written contiguously:

```python
ASC, per 128 channels -- 8 operations:
    a0 = S.vabs(...); a1 = S.vabs(...)
    S.vsts(amax_ub[group],     S.vcmax(a0, mask_low),  dist="ONEPT_B32")
    S.vsts(amax_ub[group + 1], S.vcmax(a0, mask_high), dist="ONEPT_B32")
    S.vsts(amax_ub[group + 2], S.vcmax(a1, mask_low),  dist="ONEPT_B32")
    S.vsts(amax_ub[group + 3], S.vcmax(a1, mask_high), dist="ONEPT_B32")

PTO, per 128 channels -- 3 operations:
    x = V.vcvt(V.vload(x_ub[col], size=128), "float32")
    V.vstore(V.vcmax(V.vabs(x, mask), mask, group=4), amax_ub[group])
```

This is the single most useful idea in VMI. A segmented reduce is what the
hardware does anyway — a register is organised as 8 lane groups of 32 bytes, and
reduction within a lane group is the primitive. ASC exposes it only through
`vcgmax` with a *fixed* grouping; VMI makes the segment count an argument, so it
matches the algorithm's group size instead of the register's geometry.

### The broadcast back: six operations become two

```python
ASC -- 6 operations:
    i0..i3 = four S.vld(..., dist="BRC_B32")
    q0 = S.vmul(x0, S.vsel(i0, i1, mask_low))
    q1 = S.vmul(x1, S.vsel(i2, i3, mask_low))

PTO -- 2 operations:
    inv = V.vload(inv_ub[group], size=128, stride=1, dist_mode="brc", group=4)
    q   = V.vmul(x, inv, mask)
```

The same `group=` idea, applied to a load instead of a reduce. **That symmetry is
the point**: in VMI "segmented" is a property you can request of reduce, broadcast,
load and store alike, rather than four unrelated hardware features.

### The FP8 store states its rounding

```python
PTO: V.vstore(V.vcvt(q, "float8_e4m3fn", rounding="R", saturate="SAT"), q_ub[col])
```

`SAT` matters: FP8 quantization *must* saturate rather than wrap, or a value just
above 448 becomes a small number instead of the maximum. ASC leaves it to the
instruction's default.

### Measured

**39 ASC vector operations against 19 for PTO** — the largest ratio in the ladder
after [07](07_bf16_fast_compose.md).

## A toolchain note

The fused form used here — one 128-lane vector carried through convert, segmented
reduce, divide and segmented broadcast — **did not compile** on an earlier ptoas
release, which reported `VMI-UNSUPPORTED` on `pto.vmi.group_broadcast` and forced
a bounce through Unified Buffer between the divide and the broadcast. On the pin
this repo uses it compiles and is numerically exact.
`harness/probe/vf_lane_limits.py` re-checks it, because which bodies lower is a
property of the toolchain version rather than of the hardware.

## SIMT — a reduction is a loop again, and two traps worth knowing

This is the first variant with a reduction, and it is where SIMT's shape diverges
most from the SIMD tiers. ASC reduces 32 of 64 lanes with four masked `vcmax`
calls; PTO collapses those to one `vcmax(..., group=4)`. SIMT does neither: one
thread takes one quant group and walks it serially.

```python
for g in T.Parallel(num_groups):
    amax_l[g] = E4M3_CLAMP_MIN               # seeding makes the clamp free
    for j in T.serial(group_size):
        v = T.Cast("float32", x_ub[g * group_size + j])
        amax_l[g] = T.max(amax_l[g], T.abs(v))
    inv_l[g] = E4M3_MAX / amax_l[g]
    sf_l[g]  = amax_l[g] / E4M3_MAX

for i in T.Parallel(hidden):
    q_l[i] = T.Cast("float32", x_ub[i]) * inv_l[i // group_size]
```

15 lines against ASC's 37 and PTO's 24. But read the cycle column before
concluding anything: with `group_size = 32` this puts one thread on each quant
group, so the reduction runs 4 threads out of 128, and the measured cost is an
**11.59x** ASC's simulated VF cycles — an order of magnitude, and close to the
ladder's 14.50x mean; see
[the cycle comparison](../../quant_vf_cycle_comparison.md). Short source, slow
kernel, and the two facts have the same cause.

### Two traps this variant taught

**A 2-D UB staging buffer reads the wrong tokens.** The natural-looking
`x_ub = T.alloc_shared((num_groups, group_size), ...)` with
`T.copy(X[token, 0], x_ub)` does **not** copy the token's 128 contiguous
channels. It copies the 2-D *region* `X[token:token+num_groups, 0:group_size]` —
one group's worth from each of the next four tokens. Group 0 then comes out
correct and the rest are other tokens' data, which looks exactly like a broken
reduction. Keep UB buffers **flat** and put the group structure in the index;
shared buffers accept any index arithmetic.

**Why a serial scan rather than `T.reduce_max`?** Not because the library call
fails — `T.reduce_max(dim=1)` on a `(num_groups, group_size)` fragment compiles
and gives the right answer, and `harness/probe/simt_limits.py` checks that it
still does. The scan is used because the accumulator can be **seeded** with
`E4M3_CLAMP_MIN`, which makes the clamp free, and because every later variant
needs somewhere to put the exponent arithmetic that follows the maximum. Writing
it the same way throughout the tier is worth more than one shorter line here.

### The rule that shapes every SIMT kernel here

Inside one `T.Parallel` body, a **fragment** may be touched with exactly one
structurally identical index expression. Shared and UB buffers are exempt. So
irregular, packed or data-dependent index math lives on `alloc_shared`, and
fragments carry only the clean subscripts of the `Parallel` variables. Almost
every design choice in the SIMT tier follows from that one constraint.

## What the harness checks

- scales within 1 ULP of the oracle on the NPU tiers, **exact** on the torch tier
  (pure torch computing the same quotient has no excuse for a ULP);
- FP8 values via `assert_fp8_near`;
- `assert not q.isnan().any()` — the zero row must not produce NaN. This is the
  assertion that actually catches a missing `clamp_min`, and it is why the input
  generator zeroes a row.
- the torch tier sweeps `(32,128)`, `(8,64)` and `(64,256)`.
