# per_token 07 — the bfloat16 fast path, composed

Final `per_token` variant, and the one that reaches production's actual hot loop.

```
bfloat16 input -> bfloat16 compute -> power-of-two scale -> packed UE8M0
-> FP8 e4m3 output,  at K = 256
```

## Why compute in bfloat16

A register is 256 bytes: **64 float32 lanes or 128 bfloat16**. Reducing in
bfloat16 halves the instruction count for the amax pass. For a bandwidth-bound
kernel that is a real win, and it is why production carries a separate bf16 path.

It is only legal because the scale is a power of two, so applying it is exact in
any float format. Production gates the fast path on exactly that:

```python
value_dtype = bfloat16  only if  hidden % 256 == 0
                           and  use_packed_ue8m0
                           and  round_sf
                           and  the input is bfloat16 (or requant from packed)
```

`hidden % 256 == 0` is there because the path processes 256 values per step, which
is why this variant runs at **K = 256** rather than the ladder's usual 128.

## Absolute value as a bitwise AND

For any IEEE-like format, clearing the sign bit *is* `abs`. In bfloat16 that is
`& 0x7FFF`, which runs on the integer unit:

```python
abs_x = S.vand(T.reinterpret(x, "uint16x128"), S.vdup(0x7FFF, T.uint16))
```

Cheaper than `vabs`, and the reason the fast path reinterprets to uint16 rather
than working in float.

## ASC — getting groups of 32 out of a bfloat16 vector

This is the intricate part, and it is intricate because **32 bfloat16 values are
64 bytes** — neither one 32-byte hardware lane group nor one register. Production's
answer, reproduced here:

1. `S.vld2(..., dist="DINTLV_B16")` loads 256 contiguous bfloat16 and
   *deinterleaves* them into two 128-lane vectors: evens and odds.
2. `S.vmax(abs_even, abs_odd)` gives 128 values where lane $i$ is
   $\max(x[2i], x[2i+1])$ — each lane now covers 2 original values.
3. `S.vcgmax` is the *grouped* max: it reduces within each 32-byte hardware lane
   group, which for 16-bit elements is 16 lanes, producing 8 results. Each result
   therefore covers `16 x 2 = 32` original values — exactly one quant group.
4. `S.vintlv(zero, maxima)` widens those 8 bfloat16 results to float32 (the
   zero-interleave trick from [cast_back/06](../cast_back/06_fp4_e2m1.md)), and a
   masked 8-element store puts them in `amax_ub`.

So one 256-value strip produces 8 group maxima in about five vector operations. The
float32 path needed 8 masked `vcmax` plus 8 single-element stores for the same work.

## PTO vs ASC — the widest gap in the repo

ASC needs seven operations whose only purpose is to reach a grouping the hardware
*does* have. VMI asks for the grouping it actually wants:

```python
ASC -- 7 operations, and you have to know why each is there:
    x0, x1 = S.vld2(x_ub[col], dist="DINTLV_B16")   # deinterleave 256 -> 2x128
    a0 = S.vand(T.reinterpret(x0, "uint16x128"), abs_mask)
    a1 = S.vand(T.reinterpret(x1, "uint16x128"), abs_mask)
    maxima = S.vcgmax(S.vmax(a0, a1))               # pair lanes, then reduce 16
                                                     # per hw group -> 8 results
    dense, _ = S.vintlv(zero_bf16, T.reinterpret(maxima, "bfloat16x128"))
    S.vsts(amax_ub[group], T.reinterpret(dense, "float32x64"), mask_vl8,
           dist="NORM_B32", extent=8)

PTO -- 4 operations, and they say what they mean:
    raw   = V.vload(x_ub[col], size=256)
    abs_u = V.vand(V.vinterpret_cast(raw, "uint16"), abs_mask)
    amax  = V.vcmax(abs_u, mask, group=8)           # 8 groups of 32. Done.
    V.vstore(V.vcvt(V.vinterpret_cast(amax, "bfloat16"), "float32"),
             amax_ub[group], group=8, stride=1)
```

`group=8` on a 256-lane vector means "eight independent segments of 32" — exactly
the quant group — so **no deinterleave, no pairing, no regrouping, no
re-widening**. The one remaining reinterpret is free.

And it is the *same* `group=` that [variant 01](01_raw_fp32sf.md) used for groups
of 32 in a 64-lane float32 vector. One concept, covering every width and every
element type; ASC needs a different trick for each combination.

The apply pass also stays in bfloat16, at 256 lanes, with the inverse broadcast
eight ways — safe because a power-of-two multiply is exact. The double `vcvt` on
the store is because FP8 conversion comes from float32.

Measured: **31 ASC operations against 17**.

## What is deliberately not composed here

Production additionally composes **column-major** scales with the packed UE8M0 path
(`transpose_output_sf` in `per_token_cast_asc.py` branches on exactly that). This
variant does not, because transposing *packed bytes* needs a 128-lane uint16 gather
rather than [variant 05](05_col_major_sf.md)'s 64-lane float32 one — a different
instruction sequence, not a parameter change.

All three tiers compose the same set here, so the comparison stays fair. The gap to
production is this one config, and it is named rather than glossed.

## What the harness checks

- `assert_same_bytes` on the packed scales, and `assert_fp8_near` on the values;
- **the bfloat16 reduction must pick the same power-of-two exponent** as a float32
  reduction would:
  ```python
  assert torch.equal(decode_packed_ue8m0(packed), ref_f32)
  ```
  The reduction loses mantissa bits, but the scale uses only the *exponent*, and
  bfloat16 keeps that exactly. Asserting it is what would catch a reduction that
  silently changed the exponent while still looking plausible.
- the torch tier sweeps `(32,256)` and `(8,512)`; the NPU tiers pin `K=256`.
