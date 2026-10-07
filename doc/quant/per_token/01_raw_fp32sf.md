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

## What the harness checks

- scales within 1 ULP of the oracle on the NPU tiers, **exact** on the torch tier
  (pure torch computing the same quotient has no excuse for a ULP);
- FP8 values via `assert_fp8_near`;
- `assert not q.isnan().any()` — the zero row must not produce NaN. This is the
  assertion that actually catches a missing `clamp_min`, and it is why the input
  generator zeroes a row.
- the torch tier sweeps `(32,128)`, `(8,64)` and `(64,256)`.
