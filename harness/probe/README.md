# Toolchain limit probes

Every claim this repo makes about a *toolchain* limitation is reproducible by a
script in here, tagged with the version it was measured on. Prose goes stale
silently; a script does not.

Run them after any tilelang / ptoas version bump:

```bash
python harness/probe/vf_lane_limits.py          # compile-only, ~10s
python harness/probe/vf_lane_limits.py --run    # + numerical check under the simulator
```

## Measured 2026-10-29 on tilelang 0.1.15 + ptoas 0.70

| VF body | Result |
|---|---|
| fused 128-lane `vdiv` -> grouped `vbrc`, bf16 in | compiles, PASS |
| per_token fp32 input, fused | compiles, PASS (bit-exact) |
| per_block 32x32 as 8-lane bf16 chunks | compiles |
| per_block 32x32 as 64-lane loads + `group=` reduce | compiles, PASS |

All four bodies now compile. The 8-lane conversion failed on the previous pin;
keeping it in the probe makes that toolchain improvement reproducible. The
teaching kernels still reduce at 64 or 128 lanes with segmented `group=` because
that is also what the production kernels do.

For context, on the *previous* pin (tilelang 0.1.14 + ptoas 0.1.8) the first two
rows also failed, with `VMI-UNSUPPORTED` on `pto.vmi.group_broadcast` and
`VMI-RESIDUAL-OP` respectively. Those limits are gone; kernels written against
them no longer need a workaround.

## SIMT limits, measured 2026-10-12 on tilelang 0.1.15 + CANN 9.2.0-beta.2

```bash
python harness/probe/simt_limits.py          # compile-only, ~1 min
python harness/probe/simt_limits.py --run    # + numerical check, under the simulator
```

| body inside `T.SimtVF` | Result |
|---|---|
| float32 → e2m1 via `T.copy` into an FP4 buffer | **FAILS**: `Cannot store value with 1, expected value with 2 (1 index lanes * 2 buffer elements)` |
| bfloat16 accumulator, with `T.abs` | **FAILS**: `Unresolved call ir.Op(... name="tirx.fabs" ...)` |
| bfloat16 accumulator, without `T.abs` | **FAILS**: `Ascend device compilation failed` |
| `T.abs` on float32, for contrast | compiles, correct |
| two index patterns on one **fragment** in one `T.Parallel` | **FAILS**: `StructuralEqual()(it->second.indices, indices) is false: v: (p * 2 + 1,) and (p * 2,)` |
| the same two patterns on a **shared** buffer | compiles, correct |
| `T.reduce_max(dim=1)` on `(num_groups, group_size)`, unannotated | compiles, **correct** |
| `T.copy(X[1, 0], ub)` into a `(4, 32)` shared buffer | compiles, and copies the **2-D region** `X[1:5, 0:32]` |

Three things to take from this.

**The two FP4 variants and the two bfloat16 variants have no choice.** FP4 output
has to be encoded by hand (`per_token/04`, `per_block/03`) and the bfloat16
compute path has to be float32 (`per_token/07`, `per_channel/04`). Those are not
stylistic decisions and the variant pages say so.

**The last row is a correctness trap, not a limit.** `T.copy(X[token, 0], ub)`
into a 2-D buffer copies a 2-D *region* — one row's worth from each of several
**tokens** — not the token's contiguous channels. The probe prints
`got[0][:4] = [128.0, 129.0, 130.0, 131.0]` on numbered input, which is `X[1, 0:4]`:
the first group is right and the rest are other tokens' data, so it reads as a
broken reduction. Keep UB buffers flat and put the structure in the index.

**The `reduce_max` row is a retracted claim.** An earlier draft of the SIMT tier
said that shape needs an explicit `T.annotate_layout` and writes only lane 0
without one. It does not; the probe is the regression test for the retraction, and
the wrong numbers that prompted the claim came from the row above it.
