# cast_back 03 — decode packed UE8M0 scales

New config: `use_packed_ue8m0`. The scales now arrive as one byte each, two fused
per int16 word, and the kernel has to turn an exponent byte into a float32
multiplier.

## The format

A power-of-two scale needs no mantissa, so storing the exponent alone loses
nothing. UE8M0 is exactly that — unsigned 8-bit exponent, no sign, no mantissa:

$$
\mathrm{stored}(e) = \mathrm{exponent} + 127 \qquad \mathrm{decode}(e) = 2^{e-127}
$$

One scale costs **1 byte instead of 4**. Ascend then packs two of those bytes into
one int16, because that is the narrowest type the DMA engine moves efficiently
(CUDA packs four into an int32 — the pack factor is a per-target constant, 2 here).

Note what UE8M0 *cannot* represent: zero. A byte of `0x00` decodes to `2^-127`,
not 0, which is why the quantizer clamps `amax` from below rather than letting a
zero group produce a zero scale.

## Worked example

| stored byte | decodes to |
|---|---|
| 105 | `2^-22` (what a clamped all-zero group becomes) |
| 120 | `2^-7` = `0.0078125` |
| 127 | `2^0` = `1.0` |
| 134 | `2^7` = `128.0` |

Exponents `[127, 120]` pack into the int16 word `0x787F` = `30847`, low byte first.

`harness/doc_examples.py` re-checks this table and the shift/mask equivalence
below on every run.

## ASC — decoding with no arithmetic

A float32 is `sign | exponent(8) | mantissa(23)`, so placing byte $e$ at bit 23
with a zero mantissa **is** the number `2^(e-127)`. That is a shift and a mask —
no divide, no `exp2`:

```
scale_bits = (word << shift) & 0x7F800000      -> reinterpret as float32
```

### Extracting the right byte with a per-lane shift

One int16 word holds two exponents: the low byte is the even group, the high byte
the odd one. A 64-lane strip spans exactly those two groups, so lanes 0–31 need
the low byte and lanes 32–63 the high byte — **from the same word**.

`BRC_B16` broadcasts the 16-bit word across the register; reinterpreting as uint32
gives each lane `word | word << 16`. Then:

```
shift by 23  ->  bits 0-7  (the low byte)  land on bits 23-30
shift by 15  ->  bits 8-15 (the high byte) land on bits 23-30
```

and the mask keeps only bits 23–30. So byte selection is just a **different shift
amount per lane**, built once with a select:

```python
sf_shift = S.vsel(S.vdup(23, T.int32), S.vdup(15, T.int32), mask_low)
```

A vector shift where each lane shifts by its own amount is a real instruction
here, which is what makes this work in one pass.

### Lazy vs eager decode

This is production's *lazy* path: the packed words stay in UB and are decoded
inside the multiply loop. The alternative is to decode once per token into a
float32 scratch buffer and then reuse variant 01 unchanged — simpler to read, but
it costs an extra UB write, an extra read, and a memory barrier between them.
Production keeps lazy for that reason (`use_lazy_packed_scale`).

## PTO vs ASC — an honest draw

This is the variant where VMI buys the least, and it is worth saying so rather
than manufacturing an advantage. Both backends do the identical sequence:

```python
ASC: packed = S.vld(sf_ub[strip], dist="BRC_B16")
     bits   = S.vand(S.vshl(T.reinterpret(packed, "uint32x64"), sf_shift), exp_mask)
     scale  = T.reinterpret(bits, "float32x64")

PTO: packed = V.vload(sf_ub[strip], size=128, stride=1, dist_mode="brc", group=1)
     bits   = V.vand(V.vshl(V.vinterpret_cast(packed, "uint32"), sf_shift), exp_mask)
     scale  = V.vinterpret_cast(bits, "float32")
```

Three differences, all small:

1. **Reinterpret drops the lane count.** `"uint32x64"` versus `"uint32"` — VMI
   derives it from the total bit width, so the same expression works at any width.
   This is the one real win here, and the same property that matters much more in
   the reduction variants.
2. **Mask construction.** `S.pset(32, "PAT_VL32")` names a hardware predicate
   pattern; `V.create_mask(32, size=64)` says "32 of 64 lanes".
3. **`vsel` argument order is reversed** — ASC is `S.vsel(if_true, if_false,
   mask)`, VMI is `V.vsel(mask, if_true, if_false)`. A porting hazard that
   produces wrong answers rather than an error.

Measured: 12 operations each. A genuine tie.

And one place where VMI is the **weaker** surface: it has no per-lane variable
shift. `V.vshrs` takes a scalar amount only, so where production's ASC `cast_back`
applies a shift *vector*, the PTO path computes both byte positions and selects
between them with `vcmp` + `vsel`. This variant does not hit it — a left shift by
a vector works in both — but [per_token/05](../per_token/05_col_major_sf.md)
discusses where it does.

## GPU vs NPU

`transform_sf` on the GPU is a six-line scalar macro the compiler inlines per
element. Here the same decode is a vector shift-and-mask with a per-lane shift
amount, and production carries three different strategies for it depending on the
scale layout.

## What the harness checks

- `assert_bf16_near(..., atol=0.0)` against
  `oracle.cast_back(..., packed=True)` — byte-exact.
- the two `doc_examples` checks above: the decode table, and that the in-register
  shift/mask produces the same values as the table.
