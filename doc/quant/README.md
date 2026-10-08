# The shared quantization maths

All four kernels are the same three-line idea at four granularities. This page is
the part that does not change; each kernel's README covers what its granularity
does to the code.

## Quantize

For a group $G$ of input values sharing one scale:

$$
\mathrm{amax} = \max_{i \in G} \left\lvert x_i \right\rvert
\qquad
\mathrm{sf} = \frac{\max(\mathrm{amax}, \mathrm{clamp})}{V}
\qquad
q_i = \mathrm{cast}\left(x_i \frac{V}{\mathrm{amax}}\right)
$$

$V$ is the target format's largest finite magnitude — **448** for e4m3, **6.0** for
e2m1. Dividing by `amax/V` maps the group's largest magnitude onto $V$, so the
format's whole range is used.

Note that `q` is formed with `V/amax` rather than `1/sf`. The two differ in the last
bit or two, which matters once the scale is *stored* and the cast is a separate
kernel — see [per_token/06](per_token/06_split_requant.md), where the split path
flips 75 of 131072 FP8 codes, and [per_block/05](per_block/05_split_compose.md),
where `round_sf` makes it bit-exact instead.

## Dequantize

$$
x_i = q_i \cdot \mathrm{sf}
$$

One multiply. The whole of [cast_back](cast_back/README.md) is about the *layouts*
the scale can arrive in, not the arithmetic.

## Why clamp

Without a floor, an all-zero group gives `amax = 0`, then `sf = 0` and
`q = 0/0` — NaN, from data that was perfectly valid. So `amax` is clamped from
below. The two formats do it differently, and the difference is worth knowing:

```
E4M3_CLAMP_MIN = 1e-4                 a plain small constant
E2M1_CLAMP_MIN = 6.0 * 2**-126        the smallest value whose reciprocal
                                      scale is still finite
```

`1e-4` is what the upstream tilelang reference program uses (`.clamp(1e-4)`), and
this repo matches it so the oracle and the kernels agree. It is not the smallest
*safe* floor — it is simply a value far below anything activations reach, which
also means a group whose true `amax` falls below `1e-4` gets a slightly larger
scale than it strictly needs. The e2m1 floor is derived instead: `6.0 * 2^-126` is
the smallest `amax` for which `V/amax` does not overflow to infinity.

Every kernel in the ladder plants a zero row (`randn_with_zero_row`) so the clamp
is exercised on every run rather than only on adversarial input.

## The four granularities

| kernel | one scale per | scale array | reduce along |
|---|---|---|---|
| [cast_back](cast_back/README.md) | — (scales are an input) | varies | — |
| [per_token](per_token/README.md) | 32 channels of one token | `(M, K/32)` | $K$, contiguous |
| [per_block](per_block/README.md) | a 32x32 tile | `(M/32, K/32)` | both, flattened |
| [per_channel](per_channel/README.md) | one channel, 32 tokens | `(M/32, K)` | $M$, strided |

Suggested order: `cast_back` first — it is the only kernel with no reduction, so it
teaches the vector data path before the reduce-and-broadcast machinery. Then
`per_token` for the segmented reduction, `per_block` for the lane-width rule, and
`per_channel` for what changes when the reduction axis is the slow one.

## The configs that recur

Each appears first in one variant and then composes into later ones:

| config | introduced in | what it is |
|---|---|---|
| `round_sf` | [per_token/02](per_token/02_round_sf.md) | round the scale up to a power of two, so applying it is exact |
| packed UE8M0 | [per_token/03](per_token/03_packed_ue8m0.md) | store the scale as one exponent byte — 4x less scale memory |
| FP4 e2m1 | [cast_back/05](cast_back/05_fp4_e2m1.md) | 4-bit values, two per byte, 8 magnitudes |
| column-major scales | [per_token/05](per_token/05_col_major_sf.md) | transpose the scale array for the consuming GEMM |
| split modes | [per_token/06](per_token/06_split_requant.md) | `sf_only` / `cast_only` / `requant` as compile-time flags |
| bfloat16 compute | [per_token/07](per_token/07_bf16_fast_compose.md) | 128 lanes per register instead of 64 |

## Formats

```
e4m3   1 sign | 4 exponent | 3 mantissa    max 448      8 bits
e2m1   1 sign | 2 exponent | 1 mantissa    max 6.0      4 bits, 8 magnitudes
ue8m0  0 sign | 8 exponent | 0 mantissa    a power of two, in 1 byte
bf16   1 sign | 8 exponent | 7 mantissa    float32's exponent range
```

`ue8m0` cannot represent zero — every byte decodes to a power of two — which is why
`clamp` matters and why a zero group stores the smallest exponent rather than 0.
See [cast_back/02](cast_back/02_packed_ue8m0.md) for the decode.

## Further reading

- [gpu-vs-npu.md](../gpu-vs-npu.md) — the thread model against vector registers
- [pto-vs-asc.md](../pto-vs-asc.md) — the two NPU surfaces, with measurements
- [vf-lane-widths-and-limits.md](../vf-lane-widths-and-limits.md) — register geometry
- [known-issues.md](../known-issues.md) — toolchain limits, each with a repro
