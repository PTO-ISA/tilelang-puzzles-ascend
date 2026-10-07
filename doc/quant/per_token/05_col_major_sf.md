# per_token 05 — produce column-major scales with an in-register transpose

The hardest variant in the ladder, and the reason is worth stating up front:
**a transpose inside a vector register is not a memory operation.** You cannot
restride a register. The kernel has to compute, for every lane, which source
element that lane should receive, and then gather.

## What is being transposed, and why

The scales are written as `sf_cm[group, token]` instead of `sf[token, group]`, so
that the GEMM consuming them can fetch one tile's scales contiguously with a bulk
async copy. [cast_back/07](../cast_back/07_col_major_compose.md) showed that
*consuming* this layout is free — a broadcast load does not care about stride.
Producing it is where the work is.

A strided DMA could also do it (write each token's scales down a column, stride
$M$). That is one instruction but a terrible access pattern: 4-byte elements
scattered with a large stride. Production transposes in-register and then writes
one contiguous block.

## The lane arithmetic

The kernel accumulates scales for a block of 32 tokens into `sf_dense_ub` with
shape `(32, 64)` — token-major, one padded row per token. The output wants
`(num_groups, 32)`, group-major. For output flat index $i$:

$$
\mathrm{token} = i \bmod 32 \qquad \mathrm{group} = \left\lfloor i/32 \right\rfloor \qquad \mathrm{source} = \mathrm{token} \cdot 64 + \mathrm{group}
$$

`S.vci` gives each lane its own index, so all of that is computed as a vector,
once, outside the token loop:

```python
lane  = S.vci(0, T.int32)
token = S.vand(T.reinterpret(lane, "uint32x64"), S.vdup(31, T.uint32))
group = T.reinterpret(S.vshrs(lane, 5), "uint32x64")
idx   = S.vadd(S.vmuls(token, 64), group)
```

Then one `S.vgather2(src, idx)` fetches 64 elements in the output's order. A
64-lane gather covers 64 output elements = 2 groups x 32 tokens, so a 4-group
scale array needs two gathers.

`S.vci(0, ...)` means "lane index starting at 0" — the vector equivalent of
`threadIdx.x`, and the only way to get per-lane varying data without reading
memory.

| lane | output element | source flat index |
|---|---|---|
| 0 | `out[g=0, t=0]` | 0 |
| 1 | `out[g=0, t=1]` | 64 |
| 31 | `out[g=0, t=31]` | 1984 |
| 32 | `out[g=1, t=0]` | 1 |
| 63 | `out[g=1, t=31]` | 1985 |

## PTO vs ASC — an honest draw

Both backends do the same two things — build an index vector from the lane id, then
gather — and VMI does not make either shorter:

```python
ASC: lane = S.vci(0, T.int32)
     S.vsts(idx_ub[0], S.vadd(S.vmuls(token, 64), group), dist="NORM_B32")
     vals = S.vgather2(sf_dense_ub[0, base], idx)

PTO: lane = V.vci(T.int32(0), size=64)
     V.vstore(V.vadd(V.vmul(token, V.vbrc(T.uint32(64), size=64), mask), group),
              idx_ub[0])
     vals = V.vgather(sf_dense_ub[0, base], idx, mask)
```

VMI is slightly **more** verbose: there is no scalar-operand `vmul`, so the stride
constant has to be broadcast into a vector first, and `vgather` requires an
explicit mask where `vgather2` takes one optionally.

That is the honest shape of the comparison. VMI's advantages come from `group=`
turning emulated segment behaviour into one operation, and from width being an
argument. **Neither applies to a gather**: a gather is already exactly what the
hardware does, ASC already says so directly, and there is nothing to factor out.

Production's PTO port shows the same thing — its `transpose_vectors` is
essentially a transliteration of the ASC one, which is why the port's savings came
from the reduce/broadcast/convert paths rather than from here.

Measured: 56 ASC operations against 38 — and that gap is the *rest* of the kernel
(variant 02's scale math and variant 01's broadcast), not the transpose.

### One genuine VMI regression

VMI has no per-lane variable **shift**. ASC can build a vector of shift amounts and
apply `S.vshr(v, shifts)`, which production's ASC `cast_back` uses to extract
either byte of a packed scale word. `V.vshrs` takes a scalar amount only, so
production's PTO path computes both byte positions and selects between them with
`vcmp` + `vsel`. This variant does not hit it — a left shift by a *vector* works in
both — but it is the clearest case in the quant family where VMI is the weaker
surface, and worth knowing before porting something that relies on it.

## GPU vs NPU

On a GPU this config is nearly free: you write `out_sf[g, m] = ...` and the
compiler assigns elements to threads however it likes; a transposed store is just a
different index expression, possibly with a shared-memory staging step to keep
writes coalesced. **The thread model lets any element go anywhere.** A register
lane cannot move, so on the NPU the same change costs an index computation and a
gather.

Contrast [per_block/04](../per_block/04_col_major_tma.md), where the same config is
free on *both* — because that kernel produces one scalar per tile, and a single
value has no layout.

## What the harness checks

- the output shape really is transposed:
  `assert sf_cm.shape == (k // 32, m)`;
- `sf_cm.T` is **bit-exact** against the row-major oracle (`max_ulps=0`) — a
  transpose must not perturb a value, so anything but exactness means the gather
  indices are wrong;
- FP8 values via `assert_fp8_near`;
- `M % 32 == 0` on the NPU tiers only: the transpose works a block of 32 tokens at
  a time. The torch tier is a plain `.T` and has no such constraint, which is why
  it sweeps `M=8`.
