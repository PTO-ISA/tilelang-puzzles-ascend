"""cast_back 05 (ASC) -- per-channel scales, and the broadcast disappears.

`sf_block = (32, 1)`: the scale varies along K and is constant across 32 tokens.
`sf` is (M/32, K) -- one scale per channel.

### Why this is the easy one

Every previous variant had to *construct* a scale vector, because a scale covered
32 channels while a register covers 64 lanes. Two broadcast loads and a select,
every strip.

Here the scale varies per channel, and a 64-lane float32 register covers exactly
64 consecutive channels. So the scale vector is already sitting in memory in
precisely the layout the register wants:

    scale = S.vld(sf_ub[col])          # dist defaults to NORM: contiguous

One plain contiguous load. No `BRC_B32`, no mask, no `vsel`. The broadcast
machinery that dominated variants 01-04 is simply absent.

This is the one place in the ladder where the NPU's register model fits the
problem better than the GPU's thread model. On a GPU, per-channel scales are the
*awkward* case for the matching quantize kernel, because the reduction then runs
across threads rather than within one (see `per_channel_cast_cuda.py`, which
abandons `T.Parallel` and stages partial maxima through shared memory). Here the
same granularity is the natural case.

Worth taking as the general lesson: whether a layout is convenient depends on
which axis maps onto the hardware's parallel dimension, and that answer is
different for lanes-in-a-register than it is for threads-in-a-warp.

Run:  python puzzles/asc/quant/answer/cast_back/05_per_channel_sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_bf16_near
from common.consts import BLOCK_MN

VARIANT = "asc/cast_back/05_per_channel_sf"
LANES = 64


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """Dequantize FP8 -> bfloat16 with one scale per channel per token group."""
    assert hidden % LANES == 0 and group_tokens == 32
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, group_tokens)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_m_blocks, hidden), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((hidden,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for m_block in T.serial(num_m_blocks):
                # One scale row per 32 tokens, as in variant 04.
                T.copy(Sf[m_block, 0], sf_ub)
                for row in T.serial(group_tokens):
                    token = m_block * group_tokens + row
                    T.copy(Q[token, 0], q_ub)
                    # --- BEGIN SOLUTION hint="no mask and no broadcast this time: the scale for 64 consecutive channels is 64 consecutive float32 values, so scale = S.vld(sf_ub[col]) with the default contiguous dist"
                    with T.SimdVF():
                        for strip in T.serial(hidden // LANES):
                            col = strip * LANES
                            values = S.vcvt(S.vld(q_ub[col], dist="UNPK4_B8"), T.float32)
                            # The scale is already laid out one value per lane.
                            scale = S.vld(sf_ub[col])
                            scaled = S.vmul(values, scale)
                            S.vsts(out_ub[col], S.vcvt(scaled, T.bfloat16), dist="PK_B32")
                    # --- END SOLUTION
                    T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf)
    status.assert_on_device("cast_back 05", out)
    return out


def demo_numbers() -> None:
    print("[demo] vector operations needed to build one strip's scale vector:")
    print("[demo]   variant 01 (scale per 32 channels): vld + vld + vsel = 3")
    print("[demo]   variant 05 (scale per channel)    : vld             = 1")
    print("[demo] 64 channels map one-to-one onto 64 float32 lanes, so the scale")
    print("[demo] vector is just a contiguous load -- no broadcast at all")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    assert m % BLOCK_MN == 0
    torch.manual_seed(0)
    q = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
    sf = torch.rand(m // BLOCK_MN, k) * 0.01 + 1e-4
    ref = oracle.cast_back(q, sf, (BLOCK_MN, 1), out_dtype=torch.bfloat16)
    got = launch(q.npu(), sf.npu()).cpu()
    assert_bf16_near(got, ref, f"cast_back_channel({m},{k})", atol=0.0)
    print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} matches the oracle exactly")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "cast_back", "05_per_channel_sf")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
