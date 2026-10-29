# per_token 04 — float32 input, FP4 (e2m1) output

Two independent dtype changes. One makes the kernel simpler; the other forces a
two-step conversion with a subtle rounding requirement.

## float32 input is the easy case

bfloat16 input needs an unpacking load and a widening convert:

```python
S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32)
```

float32 input is already the vector unit's compute type, so it is a plain load:

```python
S.vld(x_ub[col])
```

The trade is bandwidth: float32 input moves twice the bytes from global memory.

## FP4 output needs bfloat16 in the middle

**There is no float32 to e2m1 conversion instruction.** Asking for one is a hard
error on both backends, measured:

```
ASC: ValueError: Unsupported vcvt conversion float32->float4_e2m1fn
PTO: TypeError: ... supports packed FP4 only for bfloat16 to float4_e2m1fn
```

So the quantized values must go `float32 -> bfloat16 -> e2m1`. That is two
roundings in a row, and doing it naively biases the result: a value exactly between
two FP4 codes can be rounded *up* in the bfloat16 step and then up again, when
rounding down the second time would have been correct. Classic double rounding.

### Round to odd

The fix is to **round the intermediate to odd**: truncate to bfloat16, but force
the last mantissa bit to 1 whenever any truncated bit was nonzero. An odd
intermediate can never sit exactly on a midpoint of the final format, so the second
rounding always goes the right way.

```
v = 1.25 + 2^-20
    truncated to bf16 : 1.25        looks exactly like a midpoint
    round-to-odd      : 1.2578125   LSB set, so not a midpoint
```

`quant_max` changes too: e2m1's largest magnitude is **6.0**, not 448, and the
clamp floor becomes `6.0 * 2^-126`. See
[cast_back/05](../cast_back/05_fp4_e2m1.md) for the full 16-code table.

## ASC — bit manipulation, fusing two halves

```python
low, high = S.vdintlv(T.reinterpret(q0, "uint16x128"),
                      T.reinterpret(q1, "uint16x128"))
odd = S.vor(high, S.vmin(low, one_u16))      # set LSB if anything was dropped
S.vsts(q_ub[col], S.vcvt(T.reinterpret(odd, "bfloat16x128"), T.float4_e2m1fn),
       dist="PK4_B32")
```

`S.vdintlv` deinterleaves: given two float32 registers viewed as 16-bit lanes it
returns all the low halves in one vector and all the high halves in another. The
high halves *are* the truncated bfloat16 values. `S.vmin(low, one)` is the sticky
bit — 1 if any low bit was set, 0 otherwise.

### A toolchain gap, worked around in both tiers

The sticky bit wants `min(low, 1)`, and the scalar-operand spelling is what
production uses. On this build it does not compile:

```
error: no matching function for call to 'asc_min_scalar'
```

`asc_min_scalar` has no uint16 overload. Both files use the vector form against a
splatted `1` instead — one extra `vdup`, identical results. Recorded in
[known issues](../../known-issues.md).

## PTO vs ASC — the round-to-odd step keeps one vector

ASC's `vdintlv` is a **pairwise** operation on two registers, so its loop is built
around producing two 64-lane halves and combining them. VMI's `vunzip` splits *one*
value, and the value is already 128 lanes, so there is no pairing at all:

```python
ASC:  low, high = S.vdintlv(reinterpret(q0, "uint16x128"),
                            reinterpret(q1, "uint16x128"))
PTO:  low, high = V.vunzip(q, "uint16")
      odd = V.vinterpret_cast(V.vor(high, V.vmin(low, one_u16)), "bfloat16")
      V.vstore(V.vcvt(odd, "float4_e2m1fn", rounding="R"), q_ub[col])
```

Same three steps, but the ASC version only makes sense as a pairwise operation on
exactly two 64-lane inputs, while PTO's is a property of one value. That difference
shows up the moment the kernel wants a different width.

`V.vcvt(..., rounding="R")` also states the final rounding mode explicitly, where
ASC leaves it to the instruction default.

Measured: 37 ASC operations against 17.

## SIMT — the variant where SIMT is decisively worse

Everything above about round-to-odd describes a two-step rounding that works
*because the hardware has a float32-to-e2m1 convert*. ASC and PTO pre-round to
bfloat16 with a sticky bit so the hardware's second rounding cannot
double-round, and then convert. That is four operations in PTO.

No FP4 convert is reachable from a `SimtVF` region, so there is no second
rounding to protect, and no round-to-odd step either. Instead the SIMT file does
the **entire e2m1 encode by hand** — the torch tier's bit-exact packer
transcribed onto scalars:

```python
bits = T.reinterpret("int32", v)
sign_nib = (bits >> 28) & 0x8
exps = (bits >> 23) & 0xFF
mant = bits & 0x7FFFFF
is_sub = exps < 127                       # below 1.0 -> the subnormal row
sh = T.min(126 - exps, 31)
mant_pre = 0x400000 | (mant >> 1)
sticky_sub = T.if_then_else(is_sub, (mant & 1) | (mant_pre & ((1 << sh) - 1)), 0)
mant_eff = T.if_then_else(is_sub, mant_pre >> sh, mant)
exp_eff = T.max(exps, 126) - 126
m2 = (mant_eff >> 21) & 3
guard, lsb = m2 & 1, (m2 >> 1) & 1
sticky = sticky_sub | (mant_eff & 0x1FFFFF)
inc = guard & (T.if_then_else(sticky != 0, 1, 0) | lsb)
tmp = (((exp_eff << 2) | m2) + inc) >> 1
nibble = sign_nib | T.min(tmp, 7)         # saturate at the 6.0 code
```

Eighteen lines of encoder against PTO's four operations, and it is the right
trade only because correctness
comes first: the result is bit-exact against the oracle packer, verified over
several seeds before the kernel was ever run.

One subtlety in that code. The subnormal path shifts right by `126 - exps`, which
for a zero input is 126 — past the word width, and undefined. The torch packer
leans on its own ">= 32 gives 0" behaviour; the kernel clamps the shift to 31
instead, which is provably identical because `mant_pre < 2**23`, so every shift
of 23 or more already yields 0.

### The packing, which does go SIMT's way

Two nibbles share a byte. A thread owning one *nibble* would have to exchange
with its neighbour; a thread owning one **output byte** encodes both halves
itself and the pack needs no exchange at all:

```python
for p in T.Parallel(hidden // 2):
    inv = inv_l[p // (group_size // 2)]
    lo = e2m1_nibble(x_ub[2 * p] * inv)
    hi = e2m1_nibble(x_ub[2 * p + 1] * inv)
    q_ub[p] = T.Cast("uint8", lo | (hi << 4))
```

Indexing the shared `x_ub` twice in one body is allowed; a fragment would reject
the second pattern.

### Where the 91.28x actually goes

This variant measures **91.28x** ASC's simulated VF cycles, which is far outside
the rest of the SIMT tier (1.20x to 10.18x) and worth taking apart rather than
filing under "encoders are expensive". It splits into two independent factors,
both counted from the kernel's own trace by `tools/vf_trace_audit.py`:

| factor | measured |
|---|---|
| more instructions than its sibling SIMT variants | 243,130 against 4,036-5,828 |
| of those, pure control flow | 44,835 branches + 22,788 predicate sets |

The first factor is the encoder, and it is irreducible: the work above has to
happen somewhere, and on ASC and PTO it happens inside one hardware instruction.

**The second factor is branch divergence, and it is not inherent.** This is the
only variant in the ladder whose trace carries a large number of `SIMT_BRANCH`
and `SIMT_START_DVG` / `SIMT_END_DVG` markers, because `T.if_then_else` on a
data-dependent per-element condition lowers to a *real branch* rather than a
predicated select. Rewriting the three conditionals as mask arithmetic —
`m = -T.Cast("int32", cond)`, then `(a & m) | (b & ~m)` — changes nothing else
and keeps both arms bit-exact. Measured on this variant:

| spelling | cycles | instructions | cycles/instr | `DVG` / `BRANCH` |
|---|---:|---:|---:|---:|
| `T.if_then_else` (default) | 562,280 | 243,130 | 2.31 | 1,408 / 44,835 |
| `branch_free=True` | **269,370** | **152,260** | **1.77** | **0 / 0** |

The branch-free form issues **37% fewer instructions and runs 2.09x fewer
cycles** — **43.7x** ASC rather than 91.3x — and the instructions it drops are
the branching itself.

(`harness/probe/simt_divergence.py` is the controlled experiment behind this. It
rebuilds the encoder standalone, so its absolute counts are for the encoder
alone rather than the whole kernel.)

On a GPU this would not arise: `cond ? a : b` over cheap arithmetic gets
predicated into a select, so the natural conditional costs nothing. So about a
2.09x slice of this variant's gap is a **codegen gap, not a cost of the thread
model**, and `doc/known-issues.md` records it as such.

### The encoder is optional, and that is worth 72.29x

Can a `SimtVF` region produce FP4 directly? No, and the reason is worth knowing
before reaching for `T.Cast`. `harness/probe/simt_fp4_convert.py` tried nine
spellings — a scalar cast, a `T.vectorized(2)` pair, an fp4 fragment at 2, 4, 8
and 16 codes per thread, a `T.copy` from a float32 fragment, an explicit
`T.Cast("float4_e2m1fnx2", ...)` and a reinterpret of one code as a byte — and
all nine fail before codegen. An fp4 element is four bits, so fp4 storage is
retyped to `float4_e2m1fnx2` (one byte, two codes) and every store into it needs
two lanes, where per-thread scalar code has one. The toolchain has **no scalar
FP4 type at all**: the conversion is `vcvt(vector_f4e2m1x2, vector_bf16, ...)`,
a vector instruction over a vector-only type.

The kernel can leave the region for it, though — a `T.SimtVF` region and a
`T.SimdVF` region coexist in one kernel. That spelling lives in its own
function, `compile_hybrid_kernel`, rather than behind an `if` inside the tier's
kernel: the two differ in their output stage *and* in their output dtype, and
reading either should not mean stepping over the other:

```python
with T.SimdVF():
    for pair in T.serial(hidden // PAIR):
        S.vsts(q_ub[pair * PAIR],
               S.vcvt(S.vld(bf_ub[pair * PAIR]), T.float4_e2m1fn),
               dist="PK4_B32")
```

Three lines in place of eighteen, and **562,280 cycles down to 7,778** — 1.26x
ASC rather than 91.3x. Bit-exact against the torch packer, byte-identical to the
other two spellings, and the harness checks all three.

It is **not** the default. This tier measures what the thread model costs on its
own, and a kernel that drops into `T.SimdVF` for its hardest instruction is no
longer answering that. What it does answer — can a mixed kernel have both? — is
worth the slide it gets in the deck.

### All three spellings ship

The variant file carries all three — two spellings of the encoder inside
`compile_kernel`, and `compile_hybrid_kernel` beside it with no encoder at all:

```python
def e2m1_nibble(v, branch_free: bool = False): ...

launch(x)                      # the natural conditional -- what the ladder measures
launch(x, branch_free=True)    # the mask form -- 2.09x fewer cycles
launch(x, hw_convert=True)     # no encoder: a SimdVF vcvt -- 72.29x fewer
```

The default stays the conditional on purpose: this ladder wants the spelling a
GPU programmer reaches for first, and a measurement of *that* is more useful
than a tuned number. But the faster form is not a footnote in a probe — it is a
reference answer in the same file, and the harness **checks both**, asserting
the branch-free arm is byte-identical in nibbles and scales. Neither spelling
could land here without being verified.

Reading the two together is the point. The flag changes three lines inside one
helper, nothing about the algorithm, and nothing about the output. What it
changes is whether 64 threads that disagree about `exps < 127` proceed together
or take turns.

### Why the line table understates this

`tools/vf_lines.py` counts only what is inside the VF region, in every tier. PTO's
round-to-odd lives in a `@T.macro` outside it and SIMT's encoder is a Python
expression builder outside it, so **both are excluded** — which flatters SIMT
here by eighteen lines. Nominally this is 17 lines against ASC's 34 and PTO's
24, but that SIMT figure is not comparable to the others — add the encoder and
it is the longest of the three.

## What the harness checks

- scales within 1 ULP (0 on the torch tier);
- **the FP4 code budget**, measured rather than tolerated:
  ```python
  differing, total = int((got_v != ref_v).sum()), got_v.numel()
  assert differing / total < 0.02
  ```
  e2m1 has only 8 magnitudes, so one differing code is a factor of up to 1.5 — far
  too coarse for a value tolerance. Counting codes against the bit-exact torch
  packer is the meaningful check, and with round-to-odd in place both backends
  match it on **4096 of 4096** codes.
- the round-trip relative error is reported (about 11–12%), which is what one
  mantissa bit buys.
