# Toolchain limit probes

Every claim this repo makes about a *toolchain* limitation is reproducible by a
script in here, tagged with the version it was measured on. Prose goes stale
silently; a script does not.

Run them after any tilelang / ptoas version bump:

```bash
python harness/probe/vf_lane_limits.py          # compile-only, ~10s
python harness/probe/vf_lane_limits.py --run    # + numerical check under the simulator
python harness/probe/simt_reduce_max.py         # both reduction spellings, verified
python harness/probe/simt_fp4_convert.py        # the hand encoder vs a SimdVF convert
```

Two of them answer *performance* questions rather than "does it lower", and
their numbers are quoted in the deck and the report:

| probe | question | answer |
|---|---|---|
| `simt_reduce_max.py` | is `T.reduce_absmax` better than a thread-per-group serial scan? | **yes, 8.73x** — 8,204 cycles against 71,606, byte-identical output. The tier was re-spelled; see below |
| `simt_fp4_convert.py` | can a `SimtVF` region produce FP4, and what does the hand-written encoder cost? | **no, and 69.9x.** Nine spellings of a `T.Cast` to fp4 all fail — there is no scalar FP4 type on this chip. Leaving the region for a three-line `T.SimdVF` convert measures 7,778 cycles against 543,827, bit-exact |
| `simt_thread_count.py` | does declaring fewer SIMT threads help? | **on the shipped fold, yes: 64 threads is 0.81x of 128** (6,661 against 8,204) — it was within 4% on the serial scan this tier used to ship |

## Measured 2026-10-07 on tilelang 0.1.15 + ptoas vmi 0.1.9

| VF body | Result |
|---|---|
| fused 128-lane `vdiv` -> grouped `vbrc`, bf16 in | compiles, PASS |
| per_token fp32 input, fused | compiles, PASS (bit-exact) |
| per_block 32x32 as 8-lane bf16 chunks | **FAILS**: `VMI-LAYOUT-CONTRACT: pto.vmi.extf has no registered layout support` |
| per_block 32x32 as 64-lane loads + `group=` reduce | compiles, PASS |

The third row is the one real limit: **do not convert bf16 to f32 at 8 lanes.**
Reduce at 64 or 128 lanes with a segmented `group=` instead, which is also what
the production kernels do.

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

## A performance cliff, not a limit: `T.if_then_else` diverges

```bash
python harness/probe/simt_divergence.py          # both arms, ~16 min
```

Measured 2026-10-12 and re-measured 2026-10-14 to the digit, same pin,
`Ascend950PR_9599`, M=32 K=128. The same e2m1 encoder, with only its three
conditionals respelled; both arms bit-exact against the torch packer:

| spelling | cycles | instructions | cycles/instr | divergence markers |
|---|---:|---:|---:|---:|
| `T.if_then_else` (shipped) | 615,170 | 124,294 | 4.95 | 5,864 |
| mask arithmetic | **351,956** | 179,908 | **1.96** | **0** |

In this standalone encoder the branch-free form issues 45% **more** instructions
and still runs **1.75x fewer cycles**, because `T.if_then_else` on a
data-dependent per-element condition lowers to a real branch instead of a
predicated select.

The same comparison on the **whole variant** (`doc/known-issues.md`,
`doc/quant/per_token/04_fp32_in_fp4_out.md`) reports 562,280 against 269,370
cycles, a 2.09x gap, and there the branch-free arm issues **fewer**
instructions, 152,260 against 243,130. Both are real and they are not
comparable: a divergent branch issues every path it takes, so the conditional
arm's instruction count depends on the data, and this probe's input is not the
harness's. The mask arm, which has no branches at all, is the stable one. What
both measurements agree on is the mechanism and its sign — removing the branches
removes the divergence (5,864 markers to 0 here, 1,408 to 0 in the variant) and
the cycles follow.

This is why `simt/per_token/04` measures 91.3x ASC while the rest of the SIMT
tier sits between 1.2x and 10.2x — and why roughly a 2.09x slice of that gap is
a codegen gap rather than a cost of the thread model. See
`doc/known-issues.md`.

Both spellings now ship in that variant behind a `branch_free` flag, and the
harness asserts they are byte-identical, so this probe is the *explanation*
rather than the only place the faster form exists.
