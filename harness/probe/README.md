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
