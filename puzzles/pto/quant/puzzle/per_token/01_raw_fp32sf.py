"""per_token 01 (PTO) -- the segmented reduction, and VMI's clearest win.

Read the ASC variant first. Same three passes, same two barriers, same maths. What
changes is that the two places where ASC had to *emulate* a 32-wide group inside a
64-lane register both collapse to one operation.

### 1. The reduction: four operations become one

A group is 32 channels; a float32 register is 64 lanes. ASC's `vcmax` reduces a
whole register to one value, so covering four groups means four masked reductions
and four single-element stores:

    ASC, per 128 channels -- 8 operations:
        a0 = S.vabs(...); a1 = S.vabs(...)
        S.vsts(amax_ub[group],     S.vcmax(a0, mask_low),  dist="ONEPT_B32")
        S.vsts(amax_ub[group + 1], S.vcmax(a0, mask_high), dist="ONEPT_B32")
        S.vsts(amax_ub[group + 2], S.vcmax(a1, mask_low),  dist="ONEPT_B32")
        S.vsts(amax_ub[group + 3], S.vcmax(a1, mask_high), dist="ONEPT_B32")

VMI's reduce takes a `group=` argument meaning "this vector is N independent
segments; reduce each one". One 128-lane vector, four segments, four results
written contiguously:

    PTO, per 128 channels -- 3 operations:
        x = V.vcvt(V.vload(x_ub[col], size=128), "float32")
        V.vstore(V.vcmax(V.vabs(x, mask), mask, group=4), amax_ub[group])

This is the single most useful idea in VMI. A segmented reduce is what the
hardware does anyway -- a vector register is organised as 8 lanes of 32 bytes, and
reduction within a lane group is the primitive. ASC exposes it only through
`vcgmax` with a fixed grouping; VMI makes the segment count an argument, so it
matches the algorithm's group size instead of the register's geometry.

### 2. The broadcast back: six operations become one

Applying the scale needs the inverse of each group's scale spread over its 32
channels. ASC broadcasts each scalar and stitches with selects:

    ASC -- 6 operations:
        i0..i3 = four S.vld(..., dist="BRC_B32")
        q0 = S.vmul(x0, S.vsel(i0, i1, mask_low))
        q1 = S.vmul(x1, S.vsel(i2, i3, mask_low))

    PTO -- 2 operations:
        inv = V.vload(inv_ub[group], size=128, stride=1, dist_mode="brc", group=4)
        q   = V.vmul(x, inv, mask)

Same `group=` idea, applied to a load instead of a reduce. This symmetry is the
point: in VMI "segmented" is a property you can request of reduce, broadcast,
load and store alike, rather than four unrelated hardware features.

### 3. The FP8 store is a conversion

ASC narrows on store with `dist="PK4_B32"` after a `vcvt`. VMI converts and lets
the destination dtype imply the packing, and takes the rounding and saturation
modes as named arguments rather than leaving them implicit:

    PTO: V.vstore(V.vcvt(q, "float8_e4m3fn", rounding="R", saturate="SAT"), q_ub[col])

Being explicit matters here: FP8 quantization *must* saturate rather than wrap, or
a value just above 448 becomes a small number instead of the maximum.

### What this variant does NOT need, and once did

The fused form above -- one 128-lane vector carried through convert, segmented
reduce, divide and segmented broadcast -- **did not compile** on the previous
toolchain pin (tilelang 0.1.14 / ptoas 0.1.8), which reported `VMI-UNSUPPORTED` on
`pto.vmi.group_broadcast`. Kernels written against that pin had to bounce the
scale inverses through UB between the divide and the broadcast. On the pin this
repo uses (0.1.15 / 0.1.9) it compiles and is bit-accurate;
`common/probe/vf_lane_limits.py` re-checks it so the claim does not go stale.

### GPU vs NPU

Unchanged from the ASC variant: a GPU writes `T.reduce_absmax(..., dim=1)` and the
compiler picks the reduction strategy and the synchronisation. VMI narrows the gap
in expressiveness -- `vcmax(..., group=4)` is recognisably "reduce the last axis of
a (4, 32) view" -- but the three-pass structure, the UB staging and the barriers
are still yours to write.

Run:  python puzzles/pto/quant/answer/per_token/01_raw_fp32sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_fp32_ulps
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "pto/per_token/01_raw_fp32sf"
LANES = 64
PAIR = 128
SF_PAD = 64


# No out_idx: tilelang's automatic output allocation rejects float8_e4m3fn with
# "MemoryError: Unsupported code 10" on this toolchain (see
# common/probe/fp8_out_idx.py). launch() allocates the outputs instead.
@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize bfloat16 -> FP8 e4m3 with one FP32 scale per 32 channels."""
    assert hidden % PAIR == 0 and group_size == 32
    num_groups = hidden // group_size
    groups_per_pair = PAIR // group_size        # 4
    assert num_groups <= SF_PAD
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                # TODO: three passes. (1) per 128 channels: x =
                #       V.vcvt(V.vload(x_ub[col], size=128), 'float32'), then
                #       V.vstore(V.vcmax(V.vabs(x, mask), mask, group=4),
                #       amax_ub[group]) -- one segmented reduce, no masks per
                #       group. (2) T.simd.mem_bar('VST_VLD'); load 64 amax at
                #       once, V.vmax against the clamp, then V.vdiv both ways and
                #       store sf/inv. (3) mem_bar again; inv =
                #       V.vload(inv_ub[group], size=128, stride=1,
                #       dist_mode='brc', group=4), multiply, and
                #       V.vstore(V.vcvt(q, 'float8_e4m3fn', rounding='R',
                #       saturate='SAT'), q_ub[col])
                raise NotImplementedError("pto/per_token/01_raw_fp32sf: implement per_token_cast")
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m, hidden // CANONICAL_G), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 01", q, sf)
    return q, sf


def demo_numbers() -> None:
    print("[demo] vector operations per 128 channels, ASC vs PTO:")
    print("[demo]   reduce    ASC 8 (2 vabs, 4 masked vcmax, 4 ONEPT stores)")
    print("[demo]             PTO 3 (vload, vabs, one vcmax group=4 + store)")
    print("[demo]   broadcast ASC 6 (4 BRC loads, 2 vsel, 2 vmul)")
    print("[demo]             PTO 2 (one brc load group=4, 1 vmul)")
    print("[demo] both savings come from the same idea: `group=` makes the segment")
    print("[demo] count an argument, so it matches the algorithm's group size (32)")
    print("[demo] instead of the register's width (64).")
    print("[demo] run `python tools/vf_lines.py` for the measured counts.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_sf = oracle.per_token(x, CANONICAL_G)
    q, sf = launch(x.npu())
    assert_fp32_ulps(sf.cpu(), ref_sf, f"sf({m},{k})", max_ulps=1)
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} matches the torch oracle")
    assert not q.cpu().float().isnan().any()
    print("[check] the all-zero row clamped correctly (no NaN)")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "per_token", "01_raw_fp32sf")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
