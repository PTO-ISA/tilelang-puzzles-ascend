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
ASC needing four `BRC_B32` loads plus selects against VMI's grouped slot load and
`vbrc`. The requantize stage then reduces along $M$, where
[variant 01](01_raw_32tokens.md) showed no difference at all.

So this kernel contains one of each, which makes it a fair summary of the
comparison: **VMI collapses the segmented broadcast and changes nothing about the
elementwise reduction at the source-IR level.**

Measured: 23 ASC operations against 19.

The physical lowering is toolchain-sensitive. The previous compiler fused this
pair to `E2B_B32` and measured 721 VF cycles. PTOAS 0.70 chooses its generic
four-`BRC_B32` plus two-`VSEL` fallback for this layout and measures 1,292 cycles.
The same old compiler produced 721 cycles with both the removed one-call spelling
and the new two-call spelling, proving that the 79.2% regression belongs to the
framework upgrade rather than the API split.

## SIMT — both scale layouts become subscripts

This page's sharpest observation is that one kernel contains both cases: the
**input** scale varies along K, so it needs a broadcast (two loads and a select
on ASC, grouped slot load plus `vbrc` on PTO), while the **output** scale varies
along M, so it needs nothing. That is why PTO wins the first half of the kernel
at the source-IR level and draws the second.

In SIMT both halves are a subscript, and the difference between them is a
division:

```python
for row, c in T.Parallel(group_tokens, hidden):           # input scale: along K
    val_ub[row, c] = (T.Cast("float32", q_in_ub[row, c])
                      * sf_in_ub[row, c // in_group])

for c in T.Parallel(hidden):                              # output scale: along M
    ...
    sf_ub[c] = amax / E4M3_MAX
```

14 lines against ASC's 29 and PTO's 38. Most of that saving is the first stage,
which is the same win as [cast_back/01](../cast_back/01_e4m3_fp32sf.md).

Two implementation notes. The dequantized values live in a **shared** float32
scratch because the three stages index them differently, and only shared buffers
allow more than one index pattern. And the input scales are staged with a single
2-D `T.copy(Sf_in[mg * group_tokens, 0], sf_in_ub)` — here the 2-D region really
is what is wanted, 32 tokens by `num_in_groups` scales, which is the opposite of
the trap [per_token/01](../per_token/01_raw_fp32sf.md) describes.

### The non-bit-exact result carries over exactly

This is the one variant whose FP8 output is deliberately not bit-exact:
requantizing input that already sits on the FP8 grid produces **exact ties**,
which different implementations break differently. The harness does not loosen a
tolerance for it — it asserts that no difference exceeds one code *and* that
every differing position is a genuine midpoint.

The SIMT arm reports **the same 97 of 4096 codes, all 97 proven ties**, as ASC
and PTO do. That is worth more than passing the assertion: the count is
identical across three unrelated implementations, which shows it is a property
of where the FP8 grid falls, not of any one tier's rounding behaviour.

## What the harness checks

- scales **bit-exact** against the torch oracle;
- FP8 codes within one, **and** every difference proven to be an exact tie;
- the round-trip error after requantization is at least that after one
  quantization.
