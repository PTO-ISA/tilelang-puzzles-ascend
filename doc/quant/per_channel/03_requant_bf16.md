# per_channel 03 — requantize per-token input to per-channel scales

The kernel takes **already-quantized** input — FP8 values plus per-token scales —
dequantizes it, and requantizes along the channel axis. This is the layer-boundary
operation: one layer emits per-token scales, the next wants per-channel ones.

```
q_in (M,K) fp8  +  sf_in (M, K/32)      ->  dequantize in bfloat16
                                        ->  reduce |.| across 32 tokens
                                        ->  q (M,K) fp8  +  sf (M/32, K)
```

The dequantized values must survive between the two stages, so they go to a UB
scratch buffer with a barrier on each side — the same structure as
[per_token/06](../per_token/06_split_requant.md)'s `requant` mode.

## This variant's FP8 output is not bit-exact, and that is a result

Every other variant in the ladder matches the torch oracle to within one FP8 code
or exactly. This one has 97 differing codes out of 4096 on the default shape, and
the explanation is worth more than a loosened tolerance.

The input already sits on the FP8 grid. Dequantizing by a power of two keeps it
there, so the values entering the second quantization are exactly representable
multiples. Multiplying such a value by the new scale therefore tends to land
**exactly on a midpoint between two FP8 codes** — a tie — far more often than
random data would.

Worked through, with the actual numbers from the check:

```
value entering the second quantization : 4.5
new scale                              : 6/448
4.5 * 448/6                            = exactly 336.0
neighbouring e4m3 codes                : 320 and 352
336                                    = (320 + 352)/2     <- an exact tie
```

Both answers are correct roundings. Torch and the NPU break the tie differently,
so the code differs by one — the smallest possible difference, and not an error.

### How the harness proves it rather than tolerating it

Two assertions, and the second is the one that matters:

```python
assert worst <= 1          # no difference exceeds one FP8 code
assert ties == n_diff      # and every differing position is a genuine midpoint
```

The second recomputes the exact product and checks it against the midpoint of the
two candidate codes. A real bug — a wrong scale, a misread row, an off-by-one in
the reduction — would produce differences that are *not* midpoints, and fails this
even where it stays within one code.

I reached this by measurement, not reasoning. The first two explanations I tried,
`vdiv` precision and rounding mode, were both wrong; the tie structure is what the
data actually showed. **A tolerance would have hidden all three equally well,
which is the argument for asserting the mechanism instead of the magnitude.**

The scales, by contrast, **are** bit-exact (`max_ulps=0`), because they come from
an exponent computation with no rounding anywhere.

## Error compounds, and the check asserts the direction

```python
assert e2 >= e1, "requantizing cannot recover precision"
```

Two quantizations cannot be more accurate than one. The check reports both numbers
and asserts the inequality — a cheap guard against an implementation that
accidentally compared against the wrong reference.

## PTO vs ASC

The dequantize stage broadcasts the input scales across each group of 32 channels —
`per_token`'s pattern, where [variant 06](../per_token/06_split_requant.md) showed
ASC needing four `BRC_B32` loads plus a `vsel` against VMI's single `group=4` load.
The requantize stage then reduces along $M$, where [variant 01](01_raw_32tokens.md)
showed no difference at all.

So this kernel contains one of each, which makes it a fair summary of the
comparison: **VMI collapses the segmented broadcast and changes nothing about the
elementwise reduction.**

Measured: 23 ASC operations against 19.

## What the harness checks

- scales **bit-exact** against the torch oracle;
- FP8 codes within one, **and** every difference proven to be an exact tie;
- the round-trip error after requantization is at least that after one
  quantization.
