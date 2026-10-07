# Toolchain limit probes

Every claim this repo makes about a *toolchain* limitation is reproducible by a
script in here, tagged with the version it was measured on. Prose goes stale
silently; a script does not.

Run them after any tilelang / ptoas version bump:

```bash
python harness/probe/vf_lane_limits.py          # compile-only, ~10s
python harness/probe/vf_lane_limits.py --run    # + numerical check under the simulator
```

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
