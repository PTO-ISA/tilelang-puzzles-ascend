# per_token 03 — store the scale as one UE8M0 byte

New config: `use_packed_ue8m0`. This is the payoff of
[variant 02](02_round_sf.md): now that the scale is guaranteed to be a power of
two, its mantissa is always zero, so storing the exponent alone loses nothing.

```
variant 02:  sf is float32           4 bytes per scale
variant 03:  sf is uint8 exponent    1 byte per scale    (4x smaller)
             then packed 2-per-int16 for the DMA engine  -> (M, K/32/2) int16
```

The stored byte is `exp + 127`, float32's exponent bias. See
[cast_back/03](../cast_back/03_packed_ue8m0.md) for the decode direction and the
format's inability to represent zero.

## The convenient part

Variant 02 already computed exactly the byte we need. `biased = ceil(log2(amax/V))
+ 127` **is** the UE8M0 encoding, so this variant changes only the store.

| `amax` | biased byte | scale |
|---|---|---|
| 448.0 | 127 | `2^0` |
| 3.5 | 120 | `2^-7` |
| 1.75 | 119 | `2^-8` |

## Where the int16 packing happens

The kernel writes a flat array of **bytes**. The "two bytes per int16" layout the
public API presents is then just a reinterpretation of that same memory —
`uint8[num_groups]` viewed as `int16[num_groups/2]`, little-endian, so byte `2i`
lands in the low half of word `i`. No kernel work at all; `launch` does it with a
`.view()`.

That is worth noticing because it is easy to assume the packing needs vector work.
It only would if the pack axis were not the fastest-varying one — which is exactly
the case in [per_channel/02](../per_channel/02_round_packed_m.md), where the
scales pack along $M$ and a real `vintlv` is needed.

## ASC — narrowing 32-bit lanes to bytes

`PK4_B32` is the "pack 4x from 32-bit" store: it takes the low byte of each 32-bit
lane and writes them contiguously, so 64 lanes become 64 bytes. The reinterpret to
`"uint8x256"` tells the store the destination element width; it moves no data.

```python
S.vsts(sf_ub[0], T.reinterpret(biased, "uint8x256"), dist="PK4_B32")
```

## PTO vs ASC — a conversion instead of a reinterpret plus a store mode

Two things have to be right together in the ASC form: the reinterpret's lane count
(`uint8x256` — 64 lanes of 32 bits seen as 256 bytes) and the matching store mode
(`PK4_B32`). Get either wrong and it still compiles.

VMI converts, and the destination buffer's dtype decides the packing:

```python
V.vstore(V.vcvt(biased, "uint8"), sf_ub[0])
```

One operation, no lane bookkeeping, and the intent is legible. Same pattern as
[cast_back/02](../cast_back/02_fp32_out.md)'s store: ASC selects an instruction by
width, VMI infers it from the buffer.

Because `biased` *is* the encoding, the packed path wants the integer exponent
rather than the float32 scale — so the lane-parameterised helper from variant 02
takes a flag and returns one or the other, which is exactly the shape production's
`compute_scale` has, and only possible because the helper is reusable at all.

Measured: 44 ASC operations against 15.

## What the harness checks

- `assert_same_bytes` on the packed int16 — byte equality, since this is a pure
  bit-packing operation;
- FP8 values via `assert_fp8_near`;
- **the packed bytes decode back to the variant-02 float32 scales**:
  ```python
  assert torch.equal(decode_packed_ue8m0(packed), ref_f32)
  ```
  which is what catches a correct-looking byte written into the wrong half of the
  word.
