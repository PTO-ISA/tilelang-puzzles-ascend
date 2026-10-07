# per_channel 02 — packing along M, where the packing stops being free

`round_sf` plus packed UE8M0, as in [per_token/03](../per_token/03_packed_ue8m0.md).
The exponent arithmetic is unchanged. What changes is that **the pack axis is no
longer the fastest-varying one**, and that turns a host-side `.view()` into a real
vector instruction.

## Why this is the hard case

The scale array is `(M/32, K)`. Packing pairs token-group `2i` with token-group
`2i+1`, so the output word for channel `c` is:

```
byte 2c      =  m-group 2i   's exponent for channel c
byte 2c + 1  =  m-group 2i+1 's exponent for channel c
```

Those two bytes come from **different rows** of the scale array — `K` bytes apart
in memory, not adjacent.

| kernel | pack axis | adjacent in memory? | cost |
|---|---|---|---|
| per_token/03 | $K$ | yes | free, a `.view()` |
| per_block/02 | $K$ | yes | free, a `.view()` |
| **per_channel/02** | **$M$** | **no** | **a `vintlv` per 128 channels** |

So the kernel has to interleave two vectors of bytes. That instruction exists —
`S.vintlv` / `V.vintlv` — and it does exactly this: given two registers it returns
their element-wise interleave.

## The 256-lane trap

This is the subtlest bug in the repo, and it is worth walking through because the
cause is a property of the register rather than of the code.

A uint8 vector register is **256 lanes** — one byte per lane. The kernel wants to
interleave 128 channels at a time, because 128 inputs from each of two rows produce
256 output bytes, exactly one register. But the *load* is 256 lanes wide whatever
you intend, so loading "128 channels" from a row of `K = 128` bytes reads 128 bytes
**past the end of the row** — into the next row's scales.

The first version did that, and 75 of 256 output bytes were wrong. The failure is
quiet: the extra bytes are real data, just the wrong data, so nothing faults.

The fix is to pad the scale-byte rows so the oversized read lands in padding, and
to keep only the first of the two interleave results:

```python
a = S.vld(sf_rows_ub[0, col], dist="NORM_B8")    # reads 256 bytes, 128 wanted
b = S.vld(sf_rows_ub[1, col], dist="NORM_B8")
lo, _ = S.vintlv(a, b)                           # lo = the 256 bytes we want
S.vsts(packed_ub[col * 2], lo, dist="NORM_B8")   # hi interleaves padding: discard
```

`lo` holds the interleave of the first 128 elements of each input — precisely the
256 output bytes for these 128 channels. `hi` holds the interleave of the padding
and is thrown away.

**The transferable rule: a vector load's width is a property of the register, not
of the request.** Reading near the end of a buffer needs padding, and an in-bounds
*store* is no guarantee the *load* was in bounds.

## PTO vs ASC

Essentially identical, because an interleave is already one instruction on both:

```python
ASC: a = S.vld(sf_rows_ub[0, col], dist="NORM_B8");  lo, _ = S.vintlv(a, b)
PTO: a = V.vload(sf_rows_ub[0, col], size=256);      lo, _ = V.vintlv(a, b, byte_mask)
```

VMI states the width (`size=256`) where ASC states a distribution mode
(`NORM_B8`) — the same information, differently spelled, and arguably clearer in
VMI since it names the lane count that caused the trap above.

Measured: 28 ASC operations against 29.

## What the harness checks

- `M % 64 == 0`, with a message explaining that packing along $M$ needs an even
  number of token groups — this is why the variant's shape rule rounds `M` up to 64
  while the rest of the ladder runs at 32;
- `assert_same_bytes` on the packed output — byte equality, since this is a pure
  bit-permutation;
- FP8 values via `assert_fp8_near`;
- **`decode_packed_ue8m0_along_m(packed)` equals the float32 scales.** This is the
  assertion that catches the 256-lane bug: the wrong-row bytes were still valid
  UE8M0 exponents, so only decoding them *in the M direction* and comparing against
  the per-channel scales reveals that they came from the wrong place.
