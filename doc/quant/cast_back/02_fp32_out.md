# cast_back 02 — float32 output

New config: the **output dtype**. Nothing about the arithmetic changes; only the
store does.

In torch this is one argument:

```python
return (grouped * sf.unsqueeze(-1)).view(m, k)      # no .to(torch.bfloat16)
```

That it is a one-line change here and an instruction change on the NPU is exactly
why the torch tier exists alongside the other two.

## Worked example

Identical values to [variant 01](01_e4m3_fp32sf.md), in a wider type:

| | value |
|---|---|
| `q[0, 0:4]` | `[112, 224, -448, 56]` |
| `out[0, 0:4]` | `[1.0, 2.0, -4.0, 0.5]` as float32, **exactly** |

FP8 e4m3 carries 3 mantissa bits, so a dequantized value needs only 4 significant
bits. bfloat16 already has 8, so choosing float32 buys no accuracy for the
*quantized* values. It matters when the result feeds an accumulation that would
otherwise round repeatedly.

## ASC — the wider dtype needs the *simpler* instruction

A register is 256 bytes, and the values in its lanes are already float32. So
writing float32 is the direct case:

```python
S.vsts(out_ub[col], values, dist="NORM_B32")        # 64 lanes -> 64 x 4 bytes
```

Writing bfloat16 is the one that needs work: 64 lanes of 32 bits have to become 64
contiguous 16-bit values, so the store **packs** as it writes:

```python
S.vsts(out_ub[col], S.vcvt(values, T.bfloat16), dist="PK_B32")
```

That inverts the bandwidth intuition: the cheaper output dtype to *emit* is the
wider one. `NORM_B32` is a plain contiguous store; `PK_B32` is a narrowing one.

## PTO vs ASC

This is where VMI's handling of width stops being cosmetic. ASC picks the store
*instruction* from the output dtype, so the kernel text differs between the two
cases:

```python
ASC, bfloat16 out:  S.vsts(out_ub[col], S.vcvt(v, T.bfloat16), dist="PK_B32")
ASC, float32  out:  S.vsts(out_ub[col], v,                     dist="NORM_B32")
```

VMI infers it from the destination buffer's dtype, so the store is spelled the
same either way and only the convert appears or disappears:

```python
PTO, bfloat16 out:  V.vstore(V.vcvt(v, "bfloat16"), out_ub[col])
PTO, float32  out:  V.vstore(v,                     out_ub[col])
```

The useful consequence: a kernel parameterised over its output dtype needs no
branch in PTO, while in ASC the distribution mode has to be selected — which is
exactly what production's `store_out` helper does.

Measured: 7 vector operations for ASC, 5 for PTO.

## GPU vs NPU

On a GPU the output dtype is a property of the destination buffer and the compiler
emits whatever store that needs; there is no user-visible choice. The NPU exposes
the choice because the pack is a distribution mode on the store instruction, and
getting it wrong still compiles.

## What the harness checks

- `assert got.dtype == torch.float32` — an output that silently came back
  bfloat16 would otherwise pass the value comparison after an upcast.
- `assert_fp32_ulps(..., max_ulps=0)` — **bit-exact**, not within a ULP. The
  computation is one multiply with no intermediate rounding, so anything less than
  exact would mean the store or the convert is wrong.
