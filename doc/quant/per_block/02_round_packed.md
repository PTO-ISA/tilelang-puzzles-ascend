# per_block 02 — power-of-two scale, stored as a UE8M0 byte

Both configs from [per_token/02](../per_token/02_round_sf.md) and
[per_token/03](../per_token/03_packed_ue8m0.md), on the block layout. The
arithmetic is identical — see those pages for the ceil-log2 exponent trick and the
byte encoding — so what is worth attention here is a detail of the *layout*.

## The packing is still free, but for a different reason

`per_token` had $K/32$ scales per row, packed two-per-int16 along $K$, and the
packing was a host-side `.view()` because the pack axis was the fastest-varying
one.

`per_block` has only $K/32$ scales per *tile row*, and the kernel produces them one
at a time — one scale per tile. So the kernel writes single bytes at `Sf[i, j]`,
and the int16 pairing is again just a reinterpretation of adjacent bytes. Still no
vector work.

The case where this stops being free is
[per_channel/02](../per_channel/02_round_packed_m.md), whose scales pack along $M$
— a direction that is not adjacent in memory, and which therefore needs a real
interleave instruction.

## Only one lane matters

The scale computation here runs on a vector whose lane 0 holds the tile maximum and
whose other 63 lanes hold whatever was left in the register. That is fine — the
arithmetic is lane-wise and only lane 0 is ever stored — but it is worth noticing,
because it means the same six-operation sequence serves a 64-group `per_token` row
and a single-scale `per_block` tile without modification.

| tile `amax` | biased byte | scale |
|---|---|---|
| 448.0 | 127 | `2^0` |
| 3.5 | 120 | `2^-7` |
| 1.75 | 119 | `2^-8` |

## PTO vs ASC

The same two small differences as [per_token/03](../per_token/03_packed_ue8m0.md),
for the same reasons:

```
scale byte:  ASC  S.vsts(sf_ub[0], T.reinterpret(biased, "uint8x256"),
                         dist="PK4_B32")        # reinterpret + store mode
             PTO  V.vstore(V.vcvt(biased, "uint8"), sf_ub[0])    # one convert

exponent:    ASC reinterprets through "uint32x64" / "float32x64", so the
                 sequence is pinned to 64 lanes
             PTO reinterprets without a lane count, so the same expressions
                 serve any width
```

Neither matters much in a kernel that computes one scale per tile. They matter in
production, where the same scale arithmetic is reached from several differently
shaped paths and ASC has to repeat it per width.

Measured: 27 ASC operations against 30 — PTO is *longer* here, for the reasons in
the [overview](README.md).

## What the harness checks

- `assert_same_bytes` on the packed int16 — byte equality;
- FP8 values via `assert_fp8_near`;
- the packed bytes decode back to the variant-01 float32 scales, which is what
  catches a correct byte written into the wrong half of a word.
