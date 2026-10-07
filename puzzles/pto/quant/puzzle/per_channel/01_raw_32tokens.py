"""per_channel 01 (PTO) -- reduce along M, 128 channels at a time.

Read the ASC variant: it explains why reducing along M needs no cross-lane
reduction and no broadcast (channels sit one per lane), and why that makes this
the one kernel where the NPU's register model beats the GPU's thread model.

### PTO vs ASC

Two differences, both familiar by now:

**Width.** ASC's widening load yields 64 float32 lanes, so a 128-channel row is
two tiles. VMI reaches 128 float32 lanes with one convert, so it is one. The
reduction loop therefore runs over half as many column tiles -- the same dynamic
saving as per_block, and equally invisible to a static operation count.

**The accumulator's type is explicit.** ASC's `S.alloc_local((1,), T.float32)`
allocates "a register of float32" with the width implied by the backend. VMI
spells it out:

    acc = V.alloc_local((1,), V.vreg(128, T.float32))

`V.vreg(lanes, dtype)` is a first-class vector-register type, which is what lets
production's per_channel allocate accumulators whose width depends on whether the
kernel is in its bfloat16 or float32 mode. The ASC version has to pick the count
and the width separately and keep them in sync by hand.

Both still need the register array at all, for the same reason: **SIMD values are
immutable**, so an accumulator carried across loop iterations cannot be a plain
value. Writing `acc = V.vmax(acc, v)` fails with

    Immutable variable `acc` is used outside its defining region

Run:  python puzzles/pto/quant/answer/per_channel/01_raw_32tokens.py
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
from common.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "pto/per_channel/01_raw_32tokens"
LANES = 128


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """Quantize bfloat16 -> FP8 with one FP32 scale per channel per token group."""
    assert hidden % LANES == 0, "channels are processed 64 at a time"
    assert group_tokens == 32
    num_tokens = T.dynamic("num_tokens")
    num_groups = T.ceildiv(num_tokens, group_tokens)
    num_col_tiles = hidden // LANES

    @T.prim_func
    def per_channel_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_groups, hidden), T.float32),
    ):
        with T.Kernel(1):
            # a whole token group lives in UB, because the reduction spans it
            x_ub = T.alloc_shared((group_tokens, hidden), T.bfloat16)
            q_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((hidden,), T.float32)
            inv_ub = T.alloc_shared((hidden,), T.float32)

            for mg in T.serial(num_groups):
                T.copy(X[mg * group_tokens, 0], x_ub)
                # TODO: no cross-lane reduction here. For each 128-channel tile:
                #       acc = V.alloc_local((1,), V.vreg(128, T.float32)) seeded
                #       with V.vbrc(T.float32(E4M3_CLAMP_MIN), size=128), then
                #       loop over the 32 rows doing acc[0] = V.vmax(acc[0],
                #       V.vabs(V.vcvt(V.vload(x_ub[row, col], size=128),
                #       'float32'), mask), mask). Store sf and its inverse
                #       contiguously. After a barrier, apply with a plain
                #       V.vload(inv_ub[col], size=128) -- no broadcast.
                raise NotImplementedError("pto/per_channel/01_raw_32tokens: implement per_channel_cast")
                T.copy(q_ub, Q[mg * group_tokens, 0])
                T.copy(sf_ub, Sf[mg, 0])

    return per_channel_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m // BLOCK_MN, hidden), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_channel 01", q, sf)
    return q, sf


def demo_numbers() -> None:
    print("[demo] vector work to reduce one group, per 64 channels:")
    print("[demo]   per_token  (reduce along K): vabs + 2 masked vcmax + 2 ONEPT")
    print("[demo]                                stores, then 4 BRC loads + 2 vsel")
    print("[demo]                                to broadcast the scale back")
    print("[demo]   per_channel(reduce along M): 32 x vmax. Then a plain load.")
    print("[demo] PTO reduces 128 channels per tile where ASC reduces 64, so it")
    print("[demo] runs half as many column tiles -- a dynamic saving a static")
    print("[demo] operation count cannot see.")
    print("[demo] 64 channels map one-to-one onto 64 float32 lanes, so there is")
    print("[demo] no cross-lane reduction and no broadcast at all.")
    print("[demo] on a GPU this is the hard case: the values being combined live")
    print("[demo] in different threads, so the CUDA kernel stages partials through")
    print("[demo] shared memory and syncs. Lanes and threads give opposite answers.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    assert m % BLOCK_MN == 0 and k % LANES == 0
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_sf = oracle.per_channel(x, BLOCK_MN)
    q, sf = launch(x.npu())
    assert_fp32_ulps(sf.cpu(), ref_sf, f"sf({m},{k})", max_ulps=1)
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} matches the torch oracle")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "per_channel", "01_raw_32tokens")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
