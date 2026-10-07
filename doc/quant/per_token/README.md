# per_token — one scale per 32 channels of a token

The activation quantization granularity, and the kernel where most of the ladder's
ideas live. Do [cast_back](../cast_back/README.md) first: this is the first kernel
with a **reduction** in it, and that changes the whole shape of the code.

## What it computes

$$
\mathrm{amax}[m,g] = \max_{k \in g} \left\lvert x[m,k] \right\rvert
$$

$$
\mathrm{sf}[m,g] = \frac{\max(\mathrm{amax}[m,g], \epsilon)}{V} \qquad q[m,k] = \mathrm{cast}\left(x[m,k] \cdot \frac{V}{\mathrm{amax}[m,g]}\right)
$$

where $g$ is the group of 32 channels containing $k$, $V$ is the format's largest
finite magnitude (448 for e4m3, 6 for e2m1), and $\epsilon$ is `clamp_min`.

```
x  : (M, K)      bfloat16 or float32
q  : (M, K)      float8_e4m3fn or packed float4_e2m1fn
sf : (M, K/32)   float32, or packed UE8M0 int16
```

The scale maps each group's largest magnitude onto the top of the target range, so
every group gets the format's full precision regardless of its magnitude. That is
the entire point of block-wise quantization.

### Why `clamp_min` exists

An all-zero group would give `448/0 = Inf`, then `0 * Inf = NaN`, poisoning the
output — and a packed exponent byte of `0x00` would be a NaN scale for the
downstream GEMM. Clamping `amax` from below costs nothing and makes a zero row
harmless. Every test here uses `randn_with_zero_row`, which zeroes row 0 for
exactly this reason: on purely random data a missing clamp would never show up.

## Three passes, two barriers

The reduction is what makes this kernel structurally different from `cast_back`.
No output value can be written until its group's maximum is known, so:

```
pass 1   read x, reduce |x| per group, write amax to UB
---- memory barrier ----
pass 2   read amax, compute the scale and its inverse, write both to UB
---- memory barrier ----
pass 3   read x again, multiply by the inverse, convert, store
```

Those barriers are not optional and not inserted for you. Within a vector scope
the hardware does **not** track whether a vector load aliases an earlier vector
store to the same UB address, so without them pass 2 reads amax values pass 1 has
not yet written. This is the most common source of silent wrong answers in these
kernels.

## Variants

| # | config added | the idea |
|---|---|---|
| [01](01_raw_fp32sf.md) | raw float32 scale | **the reduction**: three passes, two barriers |
| [02](02_round_sf.md) | `round_sf` | the ceil-log2 exponent trick, integer-only |
| [03](03_packed_ue8m0.md) | `use_packed_ue8m0` | the scale as one byte |
| [04](04_fp32_in_fp4_out.md) | float32 in, FP4 out | no float32 to e2m1 instruction exists |
| [05](05_col_major_sf.md) | column-major scales | an in-register transpose: `vci` + `vgather` |
| [06](06_split_requant.md) | `sf_only` / `cast_only`, requant | trace-time modes |
| [07](07_bf16_fast_compose.md) | bfloat16 compute, composed | production's hot loop |

## Running them

```bash
python -m harness.check per_token          # all 7, all 3 tiers
python -m harness.check asc/per_token/05   # the transpose, on Ascend SIMD
```

## Where PTO's advantage is largest

This kernel is where VMI earns its keep, and `python tools/vf_lines.py` says so:
variant 01 is **39 ASC vector operations against 19**. Two places where ASC has to
emulate a 32-wide group inside a 64-lane register each collapse to one operation,
and both are the same `group=` argument — see
[01](01_raw_fp32sf.md). Variant 07 is the widest single gap in the repo.

The exception is [05](05_col_major_sf.md), a draw: a gather is already exactly
what the hardware does, so there is nothing for VMI to factor out.

## GPU vs NPU

On a GPU the reduction is a library call — `T.reduce_max(y_abs_local,
y_amax_local, dim=1)` — and the compiler decides whether it is intra-thread,
intra-warp or through shared memory, and inserts the synchronisation. Here you
choose the reduction's lane extent with a predicate, place its result in Unified
Buffer yourself, and write the barriers. Same three lines of maths.

Three further contrasts are sharpest in this kernel:

- **Layout annotations vs lane arithmetic.** The CUDA version tunes
  `T.annotate_layout(... forward_thread_fn ...)` so loads coalesce. There is no
  such knob here: the mapping from memory to lanes *is* the distribution mode you
  wrote.
- **The transpose** ([05](05_col_major_sf.md)). `out_sf[g, m] = ...` on a GPU is an
  index change; any thread can write anywhere. A register lane cannot move.
- **bfloat16 compute** ([07](07_bf16_fast_compose.md)). On a GPU, a narrower
  compute type is a dtype choice. Here it doubles the lanes per register and
  therefore changes the loop structure, the reduction strategy and even how `abs`
  is implemented.
