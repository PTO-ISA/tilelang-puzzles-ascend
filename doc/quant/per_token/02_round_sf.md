# per_token 02 — round the scale up to a power of two

New config: `round_sf`. The scale stops being $\mathrm{amax}/448$ and becomes the
smallest power of two that is at least that large:

$$
\mathrm{sf} = 2^{\left\lceil \log_2(\mathrm{amax}/V) \right\rceil}
$$

Two reasons this matters, and the second is the real one:

1. Multiplying by a power of two is **exact** — it only changes the exponent
   field, so the quantization step introduces no rounding of its own.
2. A power-of-two scale needs no mantissa, so it can be stored as a single
   exponent byte. That is [variant 03](03_packed_ue8m0.md), and it is where the
   memory saving comes from. This variant is the arithmetic that makes 03 possible.

The cost: rounding the scale *up* means the group's largest value no longer lands
on 448 but somewhere in $[224, 448]$, so up to one bit of range goes unused.

## The ceil-log2 bit trick

Kernels do not call `log2`. For a positive float32 with bit pattern
`sign | exponent(8) | mantissa(23)`:

```
bits >> 23                  gives floor(log2(v)) + 127
```

`floor` is the wrong direction — it would make the scale too small and let values
saturate past 448. Subtracting 1 before the shift fixes it:

```
biased = ((bits - 1) >> 23) + 1          ==  ceil(log2(v)) + 127
```

Checked by hand:

| $v$ | bits | `((bits-1)>>23)+1-127` | result | |
|---|---|---|---|---|
| 1.0 | `0x3F800000` | 0 | `2^0` = 1 | exact |
| 1.5 | `0x3FC00000` | 1 | `2^1` = 2 | rounded up |
| 2.0 | `0x40000000` | 1 | `2^1` = 2 | exact |
| 0.3 | `0x3E99999A` | −1 | `2^-1` = 0.5 | rounded up |

An exact power of two stays put; anything else goes up.

And the reciprocal is built by **subtracting** the exponent rather than dividing:

```
sf     = biased          << 23      ->  2^ceil_exp
sf_inv = (254 - biased)  << 23      ->  2^-ceil_exp
```

`254 - biased` works out as `127 - ceil_exp`, the negated exponent, still biased.
Negating an exponent field is exact; dividing would not be.

`harness/doc_examples.py` asserts both of these against
`math.ceil(math.log2(v))` and against `2**-exp`, so the table above cannot rot.

## ASC

Every step is a vector operation on 64 lanes at once, so 64 groups' scales are
computed in about six instructions:

```python
bits   = T.reinterpret(S.vmuls(clamped, 1.0 / 448.0), "uint32x64")
biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
sf     = T.reinterpret(S.vshls(biased, 23), "float32x64")
sf_inv = T.reinterpret(S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
                       "float32x64")
```

`T.reinterpret` is free — it renames the bits, it does not move them. The vector
unit does integer and float operations on the same registers.

### A simulator note

These shifts are why this repo defaults to `msprof op simulator` rather than
`cannsim`. `cannsim` **hangs** on `vshr`/`vshl`, so every variant from here on is
unrunnable under that backend. See [known issues](../../known-issues.md).

## PTO vs ASC — where "width is an argument" becomes structural

ASC bakes the lane count into the dtype string it reinterprets through:

```python
bits = T.reinterpret(S.vmuls(clamped, inv_qmax), "uint32x64")
sf   = T.reinterpret(S.vshls(biased, 23), "float32x64")
```

Those `x64` suffixes mean the sequence only works at 64 lanes. A 128-lane or
4-lane version of the same six operations is *different code*, so ASC kernels end
up with the scale computation written out once per width the kernel needs.

VMI derives the lane count from the total bit width, so the identical expression
works at any size — which means the scale computation can be a **reusable helper
parameterised by width**:

```python
@T.macro
def compute_scale(amax, lanes):
    mask  = V.create_mask(lanes, size=lanes)
    bits  = V.vinterpret_cast(V.vmul(clamped, inv_qmax_v, mask), "uint32")
    ...
    return sf, inv
```

Production's PTO `per_token_cast_asc.py` defines that helper once and calls it at
**4, 64, 128 and 256 lanes** across its configurations; the ASC version cannot,
and spells the arithmetic out per path. That is a structural difference — it
changes what can be factored out — and a large part of where the production port's
83 removed lines came from.

### The cost side

VMI has no scalar-operand `vsub`, so where ASC writes `S.vsub(bits, one)` with a
splatted constant, VMI needs an explicit `V.vbrc(T.uint32(1), size=lanes)`.
Several VMI calls also require a mask ASC leaves implicit. This is the ergonomic
price `tools/vf_lines.py` measures: PTO's *operation* count is lower, its *line*
count often is not.

Measured: 45 ASC operations against 14.

## What the harness checks

- scales **bit-exact** (`max_ulps=0`) against the oracle — a power of two and its
  reciprocal are both exact, so there is no room for a ULP;
- FP8 values via `assert_fp8_near`;
- **every scale has a zero mantissa**:
  ```python
  mant = sf.view(torch.int32) & 0x7FFFFF
  assert int(mant.abs().max()) == 0
  ```
  This is the assertion that distinguishes "a power of two" from "close to a power
  of two", which a value comparison alone would not catch.
