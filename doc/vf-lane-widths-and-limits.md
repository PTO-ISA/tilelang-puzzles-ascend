# Lane widths: what is legal, and why the kernels look the way they do

Almost every structural decision in these kernels traces back to one number and
one list.

> Everything here was measured on **tilelang 0.1.15 + ptoas vmi 0.1.9**,
> 2026-10-07, the versions this repo pins. Which VF bodies lower is
> **version-sensitive**, so `python common/probe/vf_lane_limits.py` re-checks the
> compile results on demand — re-run it after a toolchain bump rather than
> trusting this text.

## The number: 256 bytes

One vector register is 256 bytes. That fixes how many values one instruction
touches:

| element type | bytes | lanes per register |
|---|---:|---:|
| int8 / uint8 / float8 | 1 | **256** |
| bfloat16 / float16 / int16 | 2 | **128** |
| float32 / int32 | 4 | **64** |
| int64 | 8 | 32 |

A float32 strip is therefore 64 wide, and that is why the loops in these kernels
step by 64 (or 128, or 256) and never by a number that comes from the problem.

The register is also physically organised as **8 lane groups of 32 bytes**. That
grouping is what grouped reductions reduce over, and it is why `vcgmax` on a
bfloat16 vector produces 8 results (16 lanes each) rather than any other count.

## The list: legal vector lengths

```
{1, 2, 4, 8, 64, 128, 256}
```

Confirmed in the installed frontend at
`tilelang/ascend/language/vmi.py:73` (`_VMI_LANE_COUNTS`).

- `1, 2, 4, 8` are **compact** — fewer than 256 bytes, living in the low part of
  one register. `8` is exactly one 32-byte lane group, which is why it exists.
- `64, 128, 256` are **full** — whole multiples of a register.

**32 is absent**, and that absence shapes several kernels. 32 float32 is 128
bytes: half a register. Neither one lane group nor one register, so there is no
type for it.

This is a *frontend* allowlist, not a chip restriction and not a VMI-spec ban. The
hardware runs "half a register live" all the time — that is what a predicate is
for. The intended spelling of "32 active float32 lanes" is a 32-of-64 prefix mask
on a 64-lane vector:

```python
mask_low = S.pset(32, "PAT_VL32")        # ASC: a named hardware pattern
mask_low = V.create_mask(32, size=64)    # VMI: a count
```

which is the old `PAT_VL32` under another name.

## Why that matters: the quant group is 32

Ascend's quantization granularity is 32 channels. The natural vector width for the
data is 32, and 32 is not available. Every awkward construction in `per_token` and
`cast_back` follows from that mismatch:

| what the algorithm wants | what the hardware offers | consequence |
|---|---|---|
| reduce 32 values | reduce a whole register (64) | two masked reduces per register (ASC), or one `vcmax(..., group=2)` (VMI) |
| one scale per 32 channels | broadcast one scalar to 64 lanes | two broadcasts + a select (ASC), or one `dist_mode="brc", group=2` load (VMI) |
| a 32-wide tile row | 64 or 128 lanes | flatten the tile and reduce across it |

The whole ASC-versus-VMI story in this repo is about that table: VMI makes the
segment count an argument, so the mismatch stops being the programmer's problem.

## And 8 lanes does not rescue you

Given that 32 is unavailable, 8 looks promising — it is one lane group, and four of
them make 32. It **does not compile** for the conversion `per_block` needs:

```
VMI-LAYOUT-CONTRACT: pto.vmi.extf has no registered layout support
  operand#0=!pto.vmi.vreg<8xbf16, layout<contiguous>>
  result#0=!pto.vmi.vreg<8xf32,  layout<num_groups = 8, slots = 8>>
```

Reproduce with `common/probe/vf_lane_limits.py` (case `per_block_8lane`). So
`per_block` flattens its tile and reduces at 64 or 128 lanes, which is also what
production does.

## Measured compile results

On tilelang 0.1.15 + ptoas vmi 0.1.9, reproduced by
`common/probe/vf_lane_limits.py`:

| VF body | result |
|---|---|
| fused 128-lane `vdiv` → grouped `vbrc`, bf16 in | **ok**, numerically exact |
| per_token float32 input, fused | **ok**, bit-exact |
| per_block 32×32 as 8-lane bf16 chunks | **fail**, `VMI-LAYOUT-CONTRACT` on `extf` |
| per_block 32×32 as 64-lane loads + `group=` | **ok** |

Only the third row fails, and `per_block` is written around it. The probe exists
because this table is a property of the toolchain rather than of the hardware: a
body that does not lower today may lower after an upgrade, and one that lowers
today is not guaranteed to keep doing so. Re-run it rather than trusting the
table.

## Conversion keeps the lane count

A convert changes how many *bytes* a value occupies, never how many *values* there
are:

```
 64 × bf16  (128 B)  --vcvt-->   64 × f32  (256 B)
128 × bf16  (256 B)  --vcvt-->  128 × f32  (512 B, two registers)
```

So you cannot "load 64 bfloat16 and get 32 float32 in a smaller register". This is
why loading 128 bfloat16 values and converting them yields a *logical* 128-lane
float32 vector spanning two physical registers — a type VMI has and ASC's dtype
strings cannot name.

## Analogy: AVX-512

Hand-written VF code resembles AVX-512 intrinsics far more than it resembles a
CUDA kernel, and the resemblance is useful because the failure modes are the same.

| | this NPU | AVX-512 |
|---|---|---|
| one full register | 256 B | 64 B (`zmm`) |
| float32 per register | **64** | **16** |
| one lane group / extract | 8 float32 (32 B) | 8 float32 (`ymm`) |
| half a register | 32 float32 | 8 float32 |
| "128 float32 in one name" | logical `V<128 × f32>` (two registers) | does not exist |

Strip-mining, convert-keeps-count, and half-a-register-as-a-mask all transfer.
Where they differ: VMI has a real 128-lane logical type, AVX-512's 8-wide
(`__m256`) is first-class where VMI's 8 is a compact slot, and VMI's `group=` is
one opcode where AVX-512 needs extract-and-reduce or blend-and-set1.

`T.Parallel` on a GPU is a different layer entirely: it names one element of an
index space and lets the compiler choose the vectorisation. Inside `T.SimdVF()`
you are already past that compiler.

## Does this mean VF code has to be verbose?

No. The shortest correct per_token body on this toolchain is:

```python
with T.SimdVF():
    mask = V.create_mask(128, size=128)
    m4   = V.create_mask(4, size=4)
    x    = V.vcvt(V.vload(x_ub[0], size=128), "float32")
    amax = V.vmax(V.vcmax(V.vabs(x, mask), mask, group=4), eps4, m4)
    inv  = V.vbrc(V.vdiv(qmax4, amax, m4), size=128, group=4)
    V.vstore(V.vcvt(V.vmul(x, inv, mask), "float8_e4m3fn",
                    rounding="R", saturate="SAT"), out_ub[0])
```

Six operations for a whole 128-channel strip, and it passes. It is still vector
code — you name vectors, masks and group counts — but the width bookkeeping that
dominated the ASC spelling is gone.

What remains verbose is the *schedule*: UB staging, DMA, memory barriers and the
strip loop are all explicit, in both backends, and VMI changes none of them.
GPU-level brevity would need tilelang to emit VMI from `T.Parallel`, which it does
not yet do.
