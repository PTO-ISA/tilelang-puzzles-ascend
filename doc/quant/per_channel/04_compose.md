# per_channel 04 — the composed bfloat16 path

The final variant of the ladder. It puts together `round_sf`, packed UE8M0 along
$M$, and a **bfloat16 reduction at 128 channels per step**.

```
bfloat16 input -> bfloat16 reduction, 128 channels at a time
-> power-of-two scale -> UE8M0 packed along M -> FP8 e4m3 output
```

## Why 128 lanes is the whole point here

[Variant 01](01_raw_32tokens.md) reduced 64 channels per load and noted the
unexploited option: the reduction is elementwise, so **nothing stops it from
covering 128 channels at once** if the values stay in bfloat16. There is no segment
structure to preserve, no `vcmax` whose group size would change — just lanes
tracking their own channels.

So the loop body does the same work for twice as many channels, and a `(32, 128)`
tile needs **one column tile instead of two**.

This is the opposite situation from [per_token/07](../per_token/07_bf16_fast_compose.md),
where going to bfloat16 was intricate: there, 32 bfloat16 values are 64 bytes,
which matches no hardware grouping, so reaching groups of 32 took a deinterleave, a
pairwise max, a grouped reduce and a re-widen. Here the group *is* the lane, so
widening the vector costs nothing at all.

**The rule: changing element width is cheap when the reduction is across the
vectorized axis and expensive when it is along it.**

## Absolute value is still a mask

As in `per_token/07`, clearing the bfloat16 sign bit is `& 0x7FFF` on the integer
unit — cheaper than `vabs`, and the reason the fast path reinterprets to uint16.

## Why the exponent survives the narrower reduction

Reducing in bfloat16 discards mantissa bits, so the maximum it finds can differ
from the float32 maximum. The scale is unaffected, because it uses only the
**exponent**, and bfloat16 has float32's full 8-bit exponent field — it loses
mantissa, never range.

The harness asserts this directly rather than trusting it:

```python
assert torch.equal(decode_packed_ue8m0_along_m(packed), ref_f32)
```

`ref_f32` comes from a float32 oracle reduction. If the bfloat16 path ever picked a
different power of two, this fails — and it would be invisible to a value
comparison, since a one-exponent error in the scale is compensated almost exactly
by the quantized values.

## PTO vs ASC

VMI reaches 128 lanes by asking for them; ASC reaches them with a deinterleaving
load that returns two registers to be combined:

```python
ASC: x0, x1 = S.vld2(x_ub[row, col], dist="DINTLV_B16")   # 2 x 128 lanes
     acc = S.vmax(acc, S.vmax(S.vand(T.reinterpret(x0, "uint16x128"), abs_mask),
                              S.vand(T.reinterpret(x1, "uint16x128"), abs_mask)))

PTO: v = V.vload(x_ub[row, col], size=256)                 # 256 bfloat16
     acc = V.vmax(acc, V.vand(V.vinterpret_cast(v, "uint16"), abs_mask, m), m)
```

ASC's version works, but it covers 256 channels as two interleaved halves whose
lane-to-channel mapping has to be undone before the scales can be stored. VMI's
covers them as one vector in channel order.

Measured: **32 ASC operations against 33** — PTO is one operation *longer*, even
though its loop covers twice the channels per step and therefore runs half as many
iterations.

That is worth a second look, because it is the same trap as
[per_block](../per_block/README.md): the width advantage is real but it is a
**runtime** effect, and a count of operations *written* cannot see a count of
operations *executed*. ASC pays for its narrower vector in iterations, not in
source. Anyone judging the two surfaces by a static diff would conclude VMI gains
nothing here, and would be wrong for the right-looking reason.

## Closing the ladder

Across the 22 variants the PTO-versus-ASC picture is not uniform, and that is the
most useful thing to take from it:

| where VMI wins | why |
|---|---|
| segmented reduce and broadcast | `group=` is one operation where ASC emulates with masks and `vsel` |
| width changes | `size=` is an argument, so one helper serves 4, 64, 128, 256 lanes ([per_token/07](../per_token/07_bf16_fast_compose.md): 31 ops against 17) |
| repeated paths | the two above compound once a kernel branches over many configs |

| where VMI does not | why |
|---|---|
| gathers ([per_token/05](../per_token/05_col_major_sf.md)) | already one instruction; ASC says it directly |
| elementwise reductions ([variant 01](01_raw_32tokens.md)) | no segment structure to collapse |
| whole-vector reduces ([per_block](../per_block/README.md), and this variant) | `group=1` has nothing to factor out; VMI pays for explicit masks, and its wider vector shows up only at runtime |
| per-lane variable shifts | **VMI has no such operation**; ASC does |

Production's PTO port removed 83 lines net. The ladder shows where those lines
came from — and where they did not.

## SIMT — one half ports and wins, the other half does not port at all

This variant composes two things, and they go opposite ways. Saying so is more
useful than averaging them.

**The M-packing wins outright**, exactly as in
[variant 02](02_round_packed_m.md): a thread owning channel `c` writes two bytes,
so `V.vintlv` and the padded scratch row both disappear.

**The bfloat16 reduction does not survive the port.** The section above is clear
that 128 lanes is the whole point here — masking off the sign bit and `V.vmax`ing
raw `uint16` patterns on the integer unit, which works because for non-negative
floats the integer order matches the float order, and which doubles the lane
count. SIMT has no lane width to trade: threads are the parallelism, `T.copy`
widens bfloat16 to float32 into a fragment, and a bfloat16 cross-thread reduction
ICEs on this toolchain. So the SIMT reduction is float32.

It is numerically identical, for the reason this page already gives in "Why the
exponent survives the narrower reduction": only the **exponent** of the maximum is
used, and bfloat16 keeps exponents exactly. The harness asserts that the chosen
exponents match what float32 would pick, and the SIMT arm passes it trivially by
being float32 throughout.

17 lines against ASC's 46 and PTO's 51 — the second-largest spread in the ladder.
But the composition being free is a tilelang property, not a vector-style one, and
roughly half of what is saved here is the lane-width machinery that SIMT cannot
use rather than machinery it has improved on.

### Closing the ladder, from the fourth tier

The three-tier version of this story ends with ASC naming the machine, PTO naming
the intent, and the measurement saying PTO costs 74% of ASC's vector operations.
The SIMT tier adds a fourth point that is not further along the same line: 53% of
ASC's source lines across all 22 variants, at a simulated cycle cost that is
consistently several times ASC's and an order of magnitude on the small-group
reductions. Easiest to write and fastest to run are different axes, and
[per_token/05](../per_token/05_col_major_sf.md) is where they most clearly
disagree.

## What the harness checks

- `M % 64 == 0` (packing along $M$, as in [variant 02](02_round_packed_m.md));
- `assert_same_bytes` on the packed scales and `assert_fp8_near` on the values;
- **the bfloat16 reduction picked the same exponents as float32**, the assertion
  above, with its own failure message.
