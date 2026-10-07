"""per_channel 01 (ASC) -- reduce along M, where the NPU has the easy job.

Read puzzles/torch/quant/answer/per_channel/01_raw_32tokens.py first. One scale
per channel, shared across 32 tokens:

    amax = max(|x|) over 32 tokens, for each channel independently
    sf   = max(amax, 1e-4) / 448
    q    = cast_e4m3(x * (448 / amax))

### No cross-lane reduction, and no broadcast

Every previous quantize variant had to reduce *within* a register and then
broadcast a scalar back out. Here the reduction axis is M, and the register's
lanes hold **channels** -- 64 consecutive channels in 64 float32 lanes, one per
lane.

So reducing 32 tokens is 32 element-wise `S.vmax` operations between whole
registers. No masks, no `vcmax`, no `ONEPT` stores:

    for row in range(32):
        v = S.vabs(...)          # 64 channels of this token
        acc = S.vmax(acc, v)     # lane c now holds max over tokens so far

And the scale comes out already laid out one value per lane, so applying it is a
plain contiguous load -- the same observation as cast_back/05. The broadcast
machinery that dominates per_token simply never appears.

### The accumulator has to be a register array

`acc` is updated across loop iterations, and **SIMD values are immutable** -- a
plain value cannot be reassigned. It has to live in a register array:

    acc = S.alloc_local((1,), T.float32)
    acc[0] = S.vdup(E4M3_CLAMP_MIN, T.float32)       # also seeds the clamp
    ...
    acc[0] = S.vmax(acc[0], v)

Writing it as a plain `acc = S.vmax(acc, v)` fails with

    Immutable variable `acc` is used outside its defining region

which is a confusing message for what is really "you need a register array".
Seeding the accumulator with the clamp floor is a small trick worth copying: it
makes the clamp free rather than a separate `vmaxs` afterwards.

### GPU vs NPU -- the one case where the NPU wins

This is the hardest of the four kernels on a GPU and the easiest here, and the
reason is worth understanding because it generalises.

On a GPU, consecutive threads hold consecutive *channels* of one token. Reducing
along M means combining values held by **different threads**, so
`per_channel_cast_cuda.py` abandons `T.Parallel` entirely: it stages partial
maxima in shared memory, calls `T.sync_threads()`, then has one owner thread per
channel do the combining, with a comment about avoiding bank conflicts. It is the
only kernel in the quant family written that way.

Here the same reduction is 32 `vmax` instructions and nothing else.

The lesson: whether a reduction axis is cheap depends on which axis maps onto the
hardware's parallel dimension. Lanes-in-a-register and threads-in-a-warp give
different answers, and it is worth checking which you have before choosing a
layout.

Run:  python puzzles/asc/quant/answer/per_channel/01_raw_32tokens.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_fp32_ulps
from common.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "asc/per_channel/01_raw_32tokens"
LANES = 64


@tilelang.jit(target="ascend")
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
                # --- BEGIN SOLUTION hint="no cross-lane reduction here. For each 64-channel tile: acc = S.alloc_local((1,), T.float32) seeded with S.vdup(E4M3_CLAMP_MIN, T.float32), then loop over the 32 rows doing acc[0] = S.vmax(acc[0], S.vabs(S.vcvt(S.vld(x_ub[row, col], dist='UNPK_B16'), T.float32))). Store sf and its inverse contiguously. After a barrier, apply with a plain S.vld(inv_ub[col]) -- no broadcast."
                with T.SimdVF():
                    qmax = S.vdup(E4M3_MAX, T.float32)
                    for ct in T.serial(num_col_tiles):
                        col = ct * LANES
                        # immutable SIMD values: the accumulator is a register array
                        acc = S.alloc_local((1,), T.float32)
                        # seeding with the clamp floor makes the clamp free
                        acc[0] = S.vdup(E4M3_CLAMP_MIN, T.float32)
                        for row in T.serial(group_tokens):
                            v = S.vabs(S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"),
                                              T.float32))
                            # lane c holds the max over tokens seen so far
                            acc[0] = S.vmax(acc[0], v)
                        S.vsts(sf_ub[col], S.vdiv(acc[0], qmax))
                        S.vsts(inv_ub[col], S.vdiv(qmax, acc[0]))
                    S.mem_bar("VST_VLD")

                    for ct in T.serial(num_col_tiles):
                        col = ct * LANES
                        # the scale is already one value per lane
                        inv = S.vld(inv_ub[col])
                        for row in T.serial(group_tokens):
                            v = S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"),
                                       T.float32)
                            S.vsts(q_ub[row, col],
                                   S.vcvt(S.vmul(v, inv), T.float8_e4m3fn),
                                   dist="PK4_B32")
                # --- END SOLUTION
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
    sim.print_banner("asc", "per_channel", "01_raw_32tokens")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
