# per_token 04 — float32 input, FP4 (e2m1) output

Two independent dtype changes. One makes the kernel simpler; the other forces a
two-step conversion with a subtle rounding requirement.

## float32 input is the easy case

bfloat16 input needs an unpacking load and a widening convert:

```python
S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32)
```

float32 input is already the vector unit's compute type, so it is a plain load:

```python
S.vld(x_ub[col])
```

The trade is bandwidth: float32 input moves twice the bytes from global memory.

## FP4 output needs bfloat16 in the middle

**There is no float32 to e2m1 conversion instruction.** Asking for one is a hard
error on both backends, measured:

```
ASC: ValueError: Unsupported vcvt conversion float32->float4_e2m1fn
PTO: TypeError: ... supports packed FP4 only for bfloat16 to float4_e2m1fn
```

So the quantized values must go `float32 -> bfloat16 -> e2m1`. That is two
roundings in a row, and doing it naively biases the result: a value exactly between
two FP4 codes can be rounded *up* in the bfloat16 step and then up again, when
rounding down the second time would have been correct. Classic double rounding.

### Round to odd

The fix is to **round the intermediate to odd**: truncate to bfloat16, but force
the last mantissa bit to 1 whenever any truncated bit was nonzero. An odd
intermediate can never sit exactly on a midpoint of the final format, so the second
rounding always goes the right way.

```
v = 1.25 + 2^-20
    truncated to bf16 : 1.25        looks exactly like a midpoint
    round-to-odd      : 1.2578125   LSB set, so not a midpoint
```

`quant_max` changes too: e2m1's largest magnitude is **6.0**, not 448, and the
clamp floor becomes `6.0 * 2^-126`. See
[cast_back/05](../cast_back/05_fp4_e2m1.md) for the full 16-code table.

## ASC — bit manipulation, fusing two halves

```python
low, high = S.vdintlv(T.reinterpret(q0, "uint16x128"),
                      T.reinterpret(q1, "uint16x128"))
odd = S.vor(high, S.vmin(low, one_u16))      # set LSB if anything was dropped
S.vsts(q_ub[col], S.vcvt(T.reinterpret(odd, "bfloat16x128"), T.float4_e2m1fn),
       dist="PK4_B32")
```

`S.vdintlv` deinterleaves: given two float32 registers viewed as 16-bit lanes it
returns all the low halves in one vector and all the high halves in another. The
high halves *are* the truncated bfloat16 values. `S.vmin(low, one)` is the sticky
bit — 1 if any low bit was set, 0 otherwise.

### A toolchain gap, worked around in both tiers

The sticky bit wants `min(low, 1)`, and the scalar-operand spelling is what
production uses. On this build it does not compile:

```
error: no matching function for call to 'asc_min_scalar'
```

`asc_min_scalar` has no uint16 overload. Both files use the vector form against a
splatted `1` instead — one extra `vdup`, identical results. Recorded in
[known issues](../../known-issues.md).

## PTO vs ASC — the round-to-odd step keeps one vector

ASC's `vdintlv` is a **pairwise** operation on two registers, so its loop is built
around producing two 64-lane halves and combining them. VMI's `vunzip` splits *one*
value, and the value is already 128 lanes, so there is no pairing at all:

```python
ASC:  low, high = S.vdintlv(reinterpret(q0, "uint16x128"),
                            reinterpret(q1, "uint16x128"))
PTO:  low, high = V.vunzip(q, "uint16")
      odd = V.vinterpret_cast(V.vor(high, V.vmin(low, one_u16)), "bfloat16")
      V.vstore(V.vcvt(odd, "float4_e2m1fn", rounding="R"), q_ub[col])
```

Same three steps, but the ASC version only makes sense as a pairwise operation on
exactly two 64-lane inputs, while PTO's is a property of one value. That difference
shows up the moment the kernel wants a different width.

`V.vcvt(..., rounding="R")` also states the final rounding mode explicitly, where
ASC leaves it to the instruction default.

Measured: 37 ASC operations against 17.

## What the harness checks

- scales within 1 ULP (0 on the torch tier);
- **the FP4 code budget**, measured rather than tolerated:
  ```python
  differing, total = int((got_v != ref_v).sum()), got_v.numel()
  assert differing / total < 0.02
  ```
  e2m1 has only 8 magnitudes, so one differing code is a factor of up to 1.5 — far
  too coarse for a value tolerance. Counting codes against the bit-exact torch
  packer is the meaningful check, and with round-to-odd in place both backends
  match it on **4096 of 4096** codes.
- the round-trip relative error is reported (about 11–12%), which is what one
  mantissa bit buys.
