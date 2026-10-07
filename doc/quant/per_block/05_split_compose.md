# per_block 05 — sf_only / cast_only, and the composed kernel

The final `per_block` variant. It composes `round_sf`, packed UE8M0 and the split
modes from [variant 02](02_round_packed.md) and
[per_token/06](../per_token/06_split_requant.md), and it carries the one assertion
in the repo that is *stronger* than its `per_token` counterpart.

| mode | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| `full` | tile amax | pow2 scale | quantize |
| `sf_only` | tile amax | pow2 scale | — |
| `cast_only` | — | load `sf`, negate exponent | quantize |

## cast_only is bit-identical here, and that is the point

`per_token/06` could only promise "within one FP8 code", because its scale was an
arbitrary float32 and `1/sf` differed from `448/amax` in the last bits. Measured
there: 75 of 131072 codes flipped on bfloat16 input.

Here the scale is a power of two, so forming the reciprocal is **exact** — it
negates the exponent field and touches nothing else:

```
sf_inv = (254 - biased) << 23        # 254 - biased  ==  127 - exp
```

So the harness can assert equality of the raw bytes rather than a tolerance:

```python
assert torch.equal(q_co.view(torch.uint8), q_full.view(torch.uint8))
```

That is a much stronger statement than any value comparison, and it is only
available because of `round_sf`. **The memory saving and the bit-compatibility of
the split kernels come from the same config** — which is the practical reason
production turns `round_sf` on rather than the 4x on scale bytes alone.

## Why the split exists

In a real pipeline the scales are often needed before the values: a scheduler may
want to size an accumulator, or the same tensor may be cast twice against scales
computed once. `sf_only` and `cast_only` let one tensor be reduced once and cast as
many times as needed. Production expresses this as the same compile-time flags, on
the same kernel source.

## PTO vs ASC

Three modes means the broadcast-and-apply step is written in two of them, so the
per-mode saving from [per_token/06](../per_token/06_split_requant.md) applies twice.
But this kernel's broadcast is of a *single tile scale* to a whole vector — a
`vdupv` in ASC, a `vbrc` in VMI — and neither is shorter than the other.

Measured: 29 ASC operations against 31. PTO is longer, and the reason is the one
from the [overview](README.md): a `group=1` whole-vector reduce gives VMI's
segmented operations nothing to collapse, while VMI still pays for explicit `size=`
and mask operands. The runtime picture differs — PTO does 8 iterations where ASC
does 16 — but the static count is what the tool can measure, and it is reported as
measured.

This is the honest end of the PTO story for `per_block`: **VMI's advantages are
real but they are not universal, and this kernel is where they are thinnest.**

## The tiers are driven differently

The NPU tiers take a compile-time `mode` on one `launch` and are checked across all
three. The torch tier exposes `torch_per_block_sf_only` and
`torch_per_block_cast_only` — there is no kernel to specialise, so the split is two
functions. The harness branches on the tier for *how to call* and compares the same
quantities.

## What the harness checks

- `sf_only` produces the oracle's packed scales **byte-exactly**;
- `full` matches the oracle via `assert_fp8_near`;
- **`cast_only` is bit-identical to `full`** — the assertion above, with the
  power-of-two reciprocal as its justification;
- the packed scales decode back to the float32 scales, catching a byte written into
  the wrong half of a word.
