# per_token 06 — sf_only, cast_only, and requantization

Three related configs, all about *not* doing the whole job. Production compiles one
kernel with these as compile-time flags, and this variant does the same: `mode`
selects which passes get emitted.

| mode | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| `full` | amax | scale | quantize |
| `sf_only` | amax | scale | — |
| `cast_only` | — | `1/sf` | quantize |
| `requant` | dequantize + amax | scale | quantize |

## Why flags rather than four kernels

Because the passes are the same code. `sf_only` is the full kernel with pass 3
deleted; `cast_only` is the full kernel with pass 1 deleted and pass 2 reduced to a
reciprocal. Expressing that as Python `if` statements in the kernel builder —
evaluated at **trace time**, so they cost nothing at runtime — is how production
keeps one source for a dozen configurations.

This is the first variant where a Python-level `if` does real work, and it is worth
contrasting with a trap: `mode` is a Python string, so the condition is evaluated
while the kernel is being built and only the taken branch is emitted. A condition
on a `T.unroll` loop variable is **not** like that — see
[cast_back/07](../cast_back/07_col_major_compose.md) and
[known issues](../../known-issues.md).

## cast_only cannot reproduce the fused path exactly

With `cast_only` the kernel only has the stored scale, so it must compute `1/sf`
and multiply. The full path forms `448/amax` directly. Those differ in the last bit
or two, which flips the occasional FP8 code.

Measured over 131072 values:

| scale kind | multipliers differing | FP8 codes flipped |
|---|---|---|
| raw float32, bfloat16 input | 1453 of 4096 groups, by up to 2 ULP | 75 |
| raw float32, float32 input | same | 1 |
| power of two | none | **0** |

So this is a practical argument for `round_sf` beyond memory: **it makes the split
kernels bit-compatible with the fused one**, because a power of two and its
reciprocal are both exact.

## requant needs a scratch buffer

The new amax cannot be known until the whole group has been dequantized, so the
dequantized values have to live somewhere between the two stages. That is
`val_ub`, with a barrier on each side — the same shape as production's
`in_config.with_sf` path.

Note the third launch passes a *different* first positional: the already-quantized
tensor, not `x`.

## PTO vs ASC — the win compounds across configurations

This is where the structural advantage from [variant 02](02_round_sf.md) pays off
visibly. The kernel has four modes, and each needs the broadcast-and-apply step. In
ASC that step is four `BRC_B32` loads plus a `vsel` per 64-lane register, written
out in the dequantize stage **and** the quantize stage:

```python
ASC, twice over:
    s0 = S.vld(xsf_ub[g],     dist="BRC_B32")
    s1 = S.vld(xsf_ub[g + 1], dist="BRC_B32")
    ... S.vmul(raw, S.vsel(s0, s1, mask_low))

PTO, twice over:
    scale = V.vload(xsf_ub[g], size=128, stride=1, dist_mode="brc", group=4)
    ... V.vmul(raw, scale, mask)
```

So the duplication costs almost nothing in VMI. **This is the mechanism behind the
production port's 83 removed lines**: a single teaching variant shows a small
difference, but a kernel that branches over `round_sf`, packing, FP4, column-major,
requant and the split spells the same broadcast machinery once per path, and VMI
shrinks every one of them.

`requant` exercises both directions at once: the dequantize stage broadcasts the
*input* scales and the quantize stage broadcasts the *output* inverses, so one mode
uses the same VMI idiom twice with different data. In ASC those are two
near-identical six-operation blocks; in VMI they are two one-line loads.

Measured: 39 ASC operations against 30.

## The one place the three tiers are driven differently

The NPU tiers take a compile-time `mode` on a single `launch`. The torch tier
exposes one function per mode — `torch_sf_only`, `torch_cast_only`,
`torch_requant` — because there is no kernel to specialise. The harness branches on
the tier for *how to call*, and compares exactly the same things.

## Simulation cost

The slowest variant in the ladder, at about **62 s** per NPU tier, because it
compiles and runs four kernel modes in one check. Still well inside the 180 s
ceiling.

## What the harness checks

Three launches, each against its own oracle:

- `sf_only` scales vs `oracle.per_token` — within 1 ULP;
- `cast_only` values vs `oracle.per_token_cast_only`, **and** a report of how many
  codes differ from the fused path, which is the number tabulated above (3 of 4096
  on the default shape);
- `requant` vs `oracle.requant_per_token` — both the scales and the values.
